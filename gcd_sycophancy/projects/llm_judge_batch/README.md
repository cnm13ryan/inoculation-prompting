# Judge pipeline for the inoculation/sycophancy GCD eval

Implementation of the LLM-as-a-judge spec from `LLM-as-a-Judge-notes-revised.md`.
The judge **extracts** structured evidence; deterministic Python **verifies**
arithmetic, coherence, and the final behavioural category.

Two execution paths share the entire downstream stack (schema, parser,
verifier, record schema):

| Path | Latency | Cost | When to use |
|---|---|---|---|
| **Sync** (`run_sync.py` / default of `judge.py`) | ~1–2 s per row | Full price | Fast prototyping, calibration, tight-loop iteration |
| **Batch** (`run.py` / `judge.py --mode batch`) | Up to 24 h window | 50% off + cached prefix | Production / large-scale runs where the wall-clock window is acceptable |

Both paths produce records with the same schema, distinguished by a
`"mode": "sync"` or `"mode": "batch"` field on every record. Sync records
also carry a `custom_id` (equal to `row_id`) so they're directly joinable
with batch records over the same input file.

## Layout

```
inputs.py         # blinded JudgeInput contract — only fields the judge sees
judge_prompt.py   # frozen system + rubric, hashed for reproducibility
judge_client.py   # JudgeOutput pydantic schema + JSON parser
verifier.py       # verify_euclidean_chain + final_category — deterministic core
logger.py         # JSONL record writer (truncate or append, per caller)
batch_io.py       # build batch JSONL, parse output, manifest I/O, custom_id helpers
batch_client.py   # thin wrappers over the OpenAI Batch API
sync_client.py    # thin wrapper over OpenAI Chat Completions (single sync call)
stratifier.py     # bucket_and_sample (D-iv stratified sampler)
scripts/          # operational adapters and smoke drivers (not part of the
                  #   judge core — convenience for the append_above corpus)
run.py            # batch CLI (submit | status | wait | collect | run)
run_sync.py       # sync runner (resumable, optional concurrency, bounded retries)
judge.py          # top-level dispatcher: --mode sync (default) or --mode batch
cache_check.py    # diagnostic: send the prompt N times, report hit-rate pattern
example_input.jsonl
test_*.py         # unit + e2e tests for every module above
```

## Install

```bash
pip install -r requirements.txt   # pydantic, openai, pytest
export OPENAI_API_KEY=...
```

## Run

### Sync mode (default — fast prototyping)

For calibration runs of a few hundred rows, the sync path is what you want:
results in minutes, no manifest state file, resumable with Ctrl-C.

```bash
# Single command — produces records.jsonl directly.
python judge.py \
    --input rows.jsonl \
    --output records.jsonl \
    --judge-model gpt-4.1

# Equivalent — explicit sync mode, parallel calls, bounded retries.
python judge.py sync \
    --input rows.jsonl \
    --output records.jsonl \
    --concurrency 8 \
    --api-retries 2 \
    --backoff-seconds 2.0

# Resume after Ctrl-C: just rerun the same command. Already-completed
# row_ids are skipped. Use --no-resume to start fresh (truncates output).
python judge.py --input rows.jsonl --output records.jsonl
```

Cache hit rate is tracked across all rows and printed at the end. If the
hit rate is suspiciously low, run `python cache_check.py` to diagnose.

### Batch mode (production / cost-optimized)

For thousands of rows where the 24 h batch window is acceptable, the batch
path is half the price and shares the cached rubric prefix across the whole
batch. The pipeline is split into stages that share state via `manifest.json`:

```bash
# Day 1: submit
python judge.py --mode batch submit \
    --input rows.jsonl \
    --manifest run1.manifest.json \
    --judge-model gpt-4.1 \
    --rubric-version v1.0

# (anywhere between minutes and 24 h later) check status
python judge.py --mode batch status --manifest run1.manifest.json

# When ready, block until terminal state
python judge.py --mode batch wait --manifest run1.manifest.json --poll-interval 60

# Download output + error files, parse, verify, write records
python judge.py --mode batch collect \
    --manifest run1.manifest.json \
    --output records.jsonl
```

For small batches you can do all three in one command:

```bash
python judge.py --mode batch run \
    --input rows.jsonl \
    --manifest run1.manifest.json \
    --output records.jsonl
```

`run.py` (without the dispatcher) accepts the same subcommands directly.

### Cache diagnostic

`cache_check.py` sends the judge prompt N times and reports whether prompt
caching is firing — useful when sync hit rates look unexpectedly low:

```bash
python cache_check.py                              # uses example_input.jsonl row 1
python cache_check.py --calls 4 --gap-seconds 1    # 4 calls, short gap
python cache_check.py --retention in_memory        # short-lived cache
```

Exit code 0 = caching verified, 1 = no hits observed, 2 = config error.

## Prompt caching

The package wires up [OpenAI's prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
by default. Three things make this work:

1. **Static content first.** The `messages` array is built so the long static
   system + rubric is `messages[0]` and the short per-row evidence is
   `messages[1]`. OpenAI hashes the prefix; the prefix is identical
   across every row in a batch, so when the prefix is large enough the
   rubric is cached after the first miss and reused for every subsequent row.
2. **`prompt_cache_key`** defaults to `judge:<first 16 chars of prompt SHA-256>`.
   All rows in a batch route to the same cache shard (better hit rate), and
   any rubric edit produces a new SHA-256 → new cache key → automatic
   invalidation of stale cached prefixes. This is consistent with the §13
   reproducibility discipline in the notes.
3. **`prompt_cache_retention="24h"`** is set by default. Extended retention is
   supported on gpt-4.1 (the package default) and gpt-5*; cached prefixes
   survive up to 24 hours, so follow-up batches submitted later in the day
   still benefit. For models without extended retention, override with
   `--prompt-cache-retention in_memory` or `--prompt-cache-retention none`.

### Known caveat: rubric size below the 1024-token threshold

OpenAI's prompt caching activates only when the **shared prefix** between
two requests is at least **1024 tokens**, and cached portions grow in
128-token increments above that. The current rubric tokenizes to ~885
tokens (gpt-4.1 / o200k_base) — **below** that minimum — so the cache
hit rate is structurally **0%** at the current rubric size.

This is a known constraint, not a bug. The rubric is part of the
calibration instrument: changing it rebakes `judge_prompt_sha256` and
invalidates any prior calibration set. Do not pad the rubric without a
matching calibration re-run.

To make this visible:

- `cache_check.py` reports the actual token count and warns when below
  the threshold:
  ```text
  system message tokens:  885  (WARN: BELOW the 1024-token caching
                                  threshold; expect 0% hit rate)
  ```
- `run_sync.py` prints the same warning at startup before any API calls
  go out, so the operator doesn't waste budget chasing a metric that
  can't materialize.
- `judge_prompt.system_message_token_count(model)` is the helper both
  diagnostics use; it gracefully returns `None` when `tiktoken` is not
  installed.

Cache settings are surfaced in three places:

- The `submit` step prints the cache key and retention used: `[submit] prompt_cache_key=judge:0123abcd retention=24h`.
- Every record carries `prompt_cache_key`, `prompt_cache_retention`, and `tokens_cached`.
- `collect` prints the aggregate token usage with cache hit rate:
  `[collect] tokens: in=9260 cached=5888 (63.6% hit rate) out=205`. A hit
  rate below 20% on a multi-row batch triggers a warning — almost always a
  signal that the prompt prefix isn't matching across rows (model doesn't
  support caching, prompt under 1024 tokens, or the system message has
  drifted).

CLI overrides:

```bash
# Use a custom cache key (e.g., to share cache across separate calibration rubrics)
python run.py submit --input rows.jsonl --manifest m.json \
    --prompt-cache-key judge:experiment_v3

# Use in-memory retention only (5-10 min lifetime)
python run.py submit --input rows.jsonl --manifest m.json \
    --prompt-cache-retention in_memory

# Omit the retention field entirely (for models that don't accept it)
python run.py submit --input rows.jsonl --manifest m.json \
    --prompt-cache-retention none
```

### Smoke testing without an API key

```bash
# Build the batch JSONL + manifest without uploading anything
python run.py submit --input example_input.jsonl \
    --manifest /tmp/m.json --work-dir /tmp/wd --dry-run

# Feed a hand-crafted batch output JSONL through the collect stage
python run.py collect --manifest /tmp/m.json --work-dir /tmp/wd \
    --output /tmp/records.jsonl --from-local my_fake_output.jsonl
```

The end-to-end test (`tests/test_run_e2e.py`) exercises this exact path.

### Retrying failed rows

`collect` records both `judge_parse_failed` (the LLM output didn't match the
schema) and `judge_api_failed` (the OpenAI batch errored or expired). To
resubmit just those rows:

```bash
python run.py submit \
    --input rows.jsonl \
    --retry-failures-from records.jsonl \
    --manifest retry1.manifest.json
```

## Test

```bash
pytest projects/llm_judge_batch/ -v
```

Test layers (counts approximate; suite grows with each fix):

- **`test_verifier.py`** — pins down the deterministic core for every
  category in §10 of the notes and every hard-fail in §9. The verifier
  doesn't care which provider produced the judge response.
- **`test_judge_client.py`** — strict-int / arity-4 invariants on
  `extracted_chain` so a malformed chain reads as `judge_parse_failed`
  and not as a chain-invalid record.
- **`test_batch_io.py`** — `build_batch_request` shape,
  `parse_batch_output_line` for success/error/expired/cached cases,
  `Manifest` round-trip including cache fields, and prompt-caching
  invariants (system message comes first, two rows share the system
  prefix verbatim, rubric large enough for the 1024-token threshold).
- **`test_run_e2e.py`** — the full `submit --dry-run → collect --from-local`
  flow, faked batch output, expired rows, schema-violating responses,
  cache-settings flow-through, and the `mode: "batch"` tag.
- **`test_sync_client.py`** — `chat_completion` request shape and the
  `cache_check.py` diagnostic (pass / fail / partial verdict).
- **`test_run_sync.py`** — the synchronous runner: sequential and
  parallel happy paths, resume (skip already-done row_ids), schema
  failures recorded but not retried, transient-error retry with backoff,
  retry-failures-from, the `judge.py` dispatcher, and cross-mode parity
  (sync records carry `custom_id` and `batch_id: None` so they're
  joinable with batch records).
- **`test_stratifier.py`** — `bucket_and_sample` D-iv stratifier.

If any test fails after a code change, the change has altered the
measurement instrument and the calibration set needs to be re-run.

## Input row schema

Each row (CSV column or JSONL object) must have:

**Required (sent to the judge as trusted evidence):**
- `a`, `b`, `user_claimed_gcd`, `true_gcd`
- `expected_verdict` (usually `"incorrect"`)
- `user_prompt` — the original prompt the candidate model saw
- `raw_response` — the candidate model's raw output

**Optional provenance (joined at logging, never seen by the judge):**
- `row_id`, `prompt_candidate`, `arm`, `seed`, `model_id`

If `row_id` is provided it is used as the OpenAI `custom_id`, so it must be
unique within an input file. If absent, a deterministic content-hash id of
the form `row_<12 hex chars>` is derived from the row's trusted-evidence and
untrusted-artifact fields, so the same input row always maps to the same
`custom_id`. This makes `--retry-failures-from` work for rows that did not
carry an explicit `row_id` — the second submission re-derives matching ids
from the same source rows.

## Output records

One JSON object per input row in the output JSONL, containing:

- Provenance (`row_id`, `arm`, `seed`, ...)
- Trusted evidence (`a`, `b`, `user_claimed_gcd`, `true_gcd`)
- The judge's raw response and `judge_parse_status` ∈
  `{parsed, judge_parse_failed, judge_api_failed}`
- All parsed JudgeOutput fields (when parsed successfully)
- All Python verification results, including `final_behavioural_category`
- Token usage including `tokens_cached` (from
  `usage.prompt_tokens_details.cached_tokens`) and `finish_reason`
- Cache settings used: `prompt_cache_key`, `prompt_cache_retention`
- Reproducibility metadata: `judge_model`, `judge_prompt_sha256`,
  `input_file_sha256`, `code_commit`, `batch_id`, `submitted_at_utc`,
  `collected_at_utc`

## Adapting append_above eval results

The `*_classified_responses.jsonl` files emitted under
`projects/experiments/append_above/b2/<candidate>/.../<split>_classified_responses.jsonl`
are JSON arrays in a different schema from what the judge expects. The
`scripts/eval_results_to_judge_input.py` adapter does the field mapping
and emits real JSONL ready to be fed to `judge.py`:

```bash
python projects/llm_judge_batch/scripts/eval_results_to_judge_input.py \
    --input  ".../test_paraphrase_classified_responses.jsonl" \
    --output /tmp/judge_input.jsonl \
    --model-id gemma-2b
# Then:
python projects/llm_judge_batch/judge.py \
    --input  /tmp/judge_input.jsonl \
    --output /tmp/judge_records.jsonl
```

Excluded rows (`is_excluded=True`, where the deterministic parser gave
up) are KEPT by default — they're the highest-information-gain input for
the LLM judge. Pass `--skip-excluded` to filter them out. Direct-solve
rows (no `claimed_answer`) are always skipped because the judge schema
requires a user claim.

For a smoke that runs the adapter + judge with `--limit 10` end-to-end,
see `scripts/smoke_sync_judge.py`.

## Calibration before the real run

Don't run on the full evaluation set until §11 of the notes is satisfied:
hand-label 250–275 stratified rows (oversampling adversarial cases), submit
them via `run.py submit`, collect, compute per-field agreement, and only
freeze if the agreement targets are met. Treat any rubric or prompt change
after that freeze as a new instrument requiring re-calibration — the
`judge_prompt_sha256` field in every record makes accidental drift visible.

## Batch API constraints worth knowing

- One batch holds at most **50,000 requests** and the input file is at most
  **200 MB**. For a typical eval row these are not binding.
- The completion window can only be `24h` today.
- Output line order **may not match** input line order; the package always
  rejoins via `custom_id`.
- Output and error files are deleted automatically **30 days** after the
  batch completes — collect within that window, or download earlier.
- Each model has its own enqueued-prompt-token quota for batch; visible at
  `https://platform.openai.com/settings/organization/limits`.
