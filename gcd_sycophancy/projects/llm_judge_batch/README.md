# Judge pipeline for the inoculation/sycophancy GCD eval (OpenAI Batch API)

Implementation of the LLM-as-a-judge spec from `LLM-as-a-Judge-notes-revised.md`,
running against the [OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch).
The judge **extracts** structured evidence; deterministic Python **verifies**
arithmetic, coherence, and the final behavioural category.

## Why batch?

The Batch API costs 50% less, has separate (much higher) rate limits, and
guarantees completion within 24 hours. For a calibration run of hundreds or
thousands of rows it is the right choice. The trade-off is asynchrony — submit
and collect are different processes — which the package handles via a
**manifest** state file.

## Layout

```
inputs.py        # blinded JudgeInput contract — only fields the judge sees
judge_prompt.py  # frozen system + rubric, hashed for reproducibility
judge_client.py  # JudgeOutput pydantic schema + JSON parser (no sync LLM call)
verifier.py      # verify_euclidean_chain + final_category — the deterministic core
logger.py        # append-only JSONL writer
batch_io.py      # build batch JSONL (incl. prompt-caching), parse output, manifest I/O
batch_client.py  # thin wrappers over the OpenAI SDK
run.py           # multi-command CLI (submit | status | wait | collect | run)
tests/           # 97 tests: verifier, batch_io, prompt-caching, end-to-end
example_input.jsonl
```

## Install

```bash
pip install -r requirements.txt   # pydantic, openai, pytest
export OPENAI_API_KEY=...
```

## Run

The pipeline is split into stages that share state via `manifest.json`:

```bash
# Day 1: submit
python run.py submit \
    --input rows.jsonl \
    --manifest run1.manifest.json \
    --judge-model gpt-4.1 \
    --rubric-version v1.0

# (anywhere between minutes and 24 h later) check status
python run.py status --manifest run1.manifest.json

# When ready, block until terminal state
python run.py wait --manifest run1.manifest.json --poll-interval 60

# Download output + error files, parse, verify, write records
python run.py collect \
    --manifest run1.manifest.json \
    --output records.jsonl
```

For small batches you can do all three in one command:

```bash
python run.py run \
    --input rows.jsonl \
    --manifest run1.manifest.json \
    --output records.jsonl
```

## Prompt caching

The package wires up [OpenAI's prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
by default. Three things make this work:

1. **Static content first.** The `messages` array is built so the long static
   system + rubric (~3000+ tokens) is `messages[0]` and the short per-row
   evidence is `messages[1]`. OpenAI hashes the prefix; the prefix is identical
   across every row in a batch, so the rubric is cached after the first miss
   and reused for every subsequent row.
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
pytest tests/ -v
```

The suite has three layers:

- **`test_verifier.py`** (60 tests) pins down the deterministic core for
  every category in §10 of the notes and every hard-fail in §9. Unchanged
  from the synchronous version — the verifier doesn't care which provider
  produced the judge response.
- **`test_batch_io.py`** (32 tests) covers `build_batch_request` shape,
  `parse_batch_output_line` for success/error/expired/cached cases,
  `Manifest` round-trip including cache fields, and the prompt-caching
  invariants (system message comes first, two rows share the system
  prefix verbatim, rubric is large enough to cross the 1024-token caching
  threshold).
- **`test_run_e2e.py`** (5 tests) drives the whole `submit --dry-run →
  collect --from-local` flow, including faked OpenAI batch output, expired
  rows, schema-violating responses, and the cache-settings flow-through.

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
