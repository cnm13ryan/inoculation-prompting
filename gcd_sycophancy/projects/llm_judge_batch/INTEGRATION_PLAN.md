# Integration plan: `llm_judge_batch` as the LLM-as-judge layer

This document is the alternative to `feat/llm-judge-derivations` (PR #131).
PR #131 builds a from-scratch judge layer (`judge_derivations.py`,
~1,120 lines + 1,037 lines of tests across milestones M1–M4) inside
`gemma_gcd/scripts/`, using the **Anthropic SDK**. This plan instead
reuses the **already-complete** `projects/llm_judge_batch/` package
(14 files, ~2,000 lines, 97 passing tests, OpenAI Batch API) as the
foundation and adds only the codebase-specific glue.

**Net effect:** ~6–10 hours of engineering instead of ~33–38 hours, at
the cost of (a) committing to OpenAI as the judge provider and
(b) inheriting the package's 12-field output schema (richer than PR #131's
6-field schema; closer to `LLM-as-a-Judge-notes-revised.md` §8).

---

## Why this exists

The `gcd_sycophancy/projects/llm_judge_batch/` package (introduced as
`gemma_gcd/judge/`, then promoted to `projects/llm_judge_batch/`) is a
faithful operationalisation of `LLM-as-a-Judge-notes-revised.md`:

* §2 *judge extracts, Python verifies* → `judge_prompt.JUDGE_RUBRIC`
  explicitly forbids the LLM from doing arithmetic;
  `verifier.verify_euclidean_chain` does the math.
* §4 *blinded input contract* → `inputs.JudgeInput` +
  `ALLOWED_JUDGE_FIELDS` enforces the blinding.
* §5.1–§5.5 *observables* → `judge_client.JudgeOutput` matches the §8
  JSON schema field-for-field.
* §5.6 *cross-field coherence as a deterministic function* →
  `verifier.coherence_class`.
* §9 *hard-fails* + §10 *composite categories* → `verifier.final_category`
  branch order matches the §10 priority.
* §12 *prompt-injection protection* → in `JUDGE_SYSTEM`.
* §13 *reproducibility logging* → `Manifest` records `prompt_sha256`,
  `code_commit`, `batch_id`, `submitted_at_utc`, `collected_at_utc`,
  per-row token usage incl. cached.

The package also implements OpenAI's Batch API end-to-end: prompt
caching with cache-key derivation from prompt SHA-256, manifest-based
state file linking submit/wait/collect, retry-failed-rows flow,
end-to-end smoke testing without an API key.

What it doesn't have: a bridge to the `prereg_problem_level_data.csv`
schema produced by this codebase's training/eval pipeline. That bridge
is what this plan builds.

---

## Comparison: PR #131 vs. this plan

| Aspect | PR #131 (`feat/llm-judge-derivations`) | This plan (`feat/llm-judge-via-batch-package`) |
|---|---|---|
| **Provider** | Anthropic (claude-sonnet-4-6) | OpenAI (gpt-4.1 default; gpt-4o-mini cheap-mode) |
| **Mode** | Sync at M4; Batch planned for M8 | Batch only (the package's entire architecture) |
| **Output schema** | 6 fields (chain, status, derivation_final, faithfulness_score, outlier_field, reasoning) | 12 fields (verdict_tag_status/value, semantic_stance, answer_tag_status/value/prose, answer_commitment_class, chain_extraction_status, extracted_chain, derivation_answer_class, degeneracy_class, confidence, short_rationale) — closer to §8 of the notes |
| **Final categories** | None (downstream consumes faithfulness + outlier directly) | `verifier.final_category` returns `genuine / surface_only / deep_sycophancy / sycophantic_agreement / math_correction_format_failure / degenerate_avoidance / ambiguous` per §10 |
| **Status** | M1–M4 complete (~2,150 lines on branch); M5–M10 pending | Package complete (~2,000 lines, 97 tests); only adapters to write |
| **Engineering remaining** | ~33–38h (M5–M10 per plan) | ~6–10h (adapters + integration + run-book) |
| **Calibration** | M5–M7 (~13h, plus 5h human labelling) | Same 5h human labelling; ~2h of glue to drive it through `run.py submit` |
| **Cost (per panel run, D-iv)** | $384 (Sonnet, batch+cache) | $30–80 (gpt-4o-mini batch+cache) or $120–250 (gpt-4.1) |
| **Reproducibility hash** | `judge_prompt_sha256` planned | Already implemented (`prompt_hash()` at module load) |

The two implementations are not mutually exclusive. They could run
side-by-side as cross-judge-agreement diagnostics — at the cost of
maintaining two parallel codepaths. That's a separate decision.

---

## What's missing — the integration glue

The `llm_judge_batch/` package reads JSONL with these columns:

```
required:  a, b, user_claimed_gcd, true_gcd,
           expected_verdict, user_prompt, raw_response
optional:  row_id, prompt_candidate, arm, seed, model_id
```

The codebase produces `prereg_problem_level_data.csv` with these columns:

```
pair_a, pair_b, claimed_answer, true_answer,
prompt_family, evaluation_set_name, prompt, response,
arm_id, arm_slug, seed, problem_id, cluster_id,
... + 40 more parser/diagnostic columns
```

Every required field of the judge is recoverable from the CSV with
column renames, plus one derivation rule:

| Judge field | CSV source | Notes |
|---|---|---|
| `a` | `pair_a` | direct |
| `b` | `pair_b` | direct |
| `user_claimed_gcd` | `claimed_answer` | cast to int (CSV stores as float) |
| `true_gcd` | `true_answer` | direct |
| `expected_verdict` | derived from `prompt_family` | `incorrect_confirmation → "incorrect"`; `correct_confirmation → "correct"`; skip `direct_solve` (no verdict expected) |
| `user_prompt` | `prompt` | direct |
| `raw_response` | `response` | direct |
| `row_id` (optional) | composed `(candidate, seed, cluster_id, eval_set)` | ensures uniqueness within a batch |
| `prompt_candidate` | candidate slug from CSV path | judge never sees it |
| `arm` | `arm_slug` (`"inoculation_prompting"` for ipb) | judge never sees it |
| `seed` | `seed` | judge never sees it |
| `model_id` | composed from candidate + seed + checkpoint timestamp | judge never sees it |

That's the only missing piece on the input side.

On the output side the judge writes a records JSONL with one row per
input row, containing all the trusted-evidence echoes plus the parsed
`JudgeOutput` fields plus the Python-verified outcomes. Merging these
records back into a sidecar CSV alongside `prereg_problem_level_data.csv`
mirrors the approach PR #131 takes (see `JUDGE_COST_ESTIMATE.md` §1
*"Sidecar over in-place"*).

---

## Concrete deliverables for this branch

### 1. Two adapter scripts (~80 lines total)

**`projects/llm_judge_batch/adapters/from_prereg_csv.py`** (~40 lines).
Reads one or more `prereg_problem_level_data.csv` files, applies the
filter (arms / eval-sets / seeds / prompt_family), renames columns,
derives `expected_verdict`, writes JSONL in the judge's input format.

```bash
python adapters/from_prereg_csv.py \
    --panel-root experiments/append_above/b2 \
    --arms 1 2 \
    --seeds 0 1 2 3 \
    --eval-sets confirmatory paraphrase same_domain_extrapolation \
    --output /tmp/append_above_judge_input.jsonl
```

**`projects/llm_judge_batch/adapters/to_sidecar_csv.py`** (~40 lines).
Reads a `records.jsonl` produced by `run.py collect`, joins to the
original CSV by composed `row_id`, writes
`prereg_problem_level_data.with_judge.csv` adding `judge_*` columns
(prefixed to avoid colliding with the existing `parsed_*`, `strict_*`,
`lenient_*`, `semantic_*` columns).

The sidecar columns:

```
judge_verdict_tag_status, judge_verdict_tag_value,
judge_semantic_stance,
judge_answer_tag_status, judge_answer_tag_value, judge_prose_answer_value,
judge_answer_commitment_class,
judge_chain_extraction_status, judge_extracted_chain (JSON-encoded),
judge_derivation_answer_class,
judge_degeneracy_class,
judge_confidence,
judge_short_rationale,

# Python verifier fields (computed by verifier.final_category)
judge_python_chain_valid, judge_python_chain_final,
judge_stance_direction, judge_answer_direction, judge_derivation_direction,
judge_coherence_class,
judge_final_behavioural_category,

# Reproducibility provenance
judge_model, judge_prompt_sha256, judge_batch_id,
judge_submitted_at_utc, judge_collected_at_utc,
judge_tokens_in, judge_tokens_cached, judge_tokens_out
```

### 2. Optional stratified pre-filter

PR #131's `bucket_and_sample` (D-iv coverage tier) is not in
`llm_judge_batch/`. Two options:

* **Inline in the adapter** (~15 extra lines). For each candidate's
  CSV, classify each row using `analyze_derivation_chain.verify_response`
  (already merged via PR #130), then sample 100 % of `derivation_other`
  + `derivation_unparseable` and 25 % of the deterministic-OK buckets
  with a fixed seed. Implements the D-iv tier described in
  `JUDGE_COST_ESTIMATE.md §3`.
* **Skip stratification entirely** (D-i, every row). Simpler;
  ~$80 vs $30 on gpt-4o-mini for confirmatory ipb. Probably worth
  the extra $50 to keep the adapter trivial.

Recommendation: build inline D-iv stratification, but make
`--coverage-tier d_i` the default to keep the adapter simple. Promote
`d_iv` to default after the first calibrated run confirms the
verifier's bucket boundaries are reliable on this data.

### 3. Run-book in `projects/llm_judge_batch/RUN_BOOK.md`

End-to-end recipe:

```
1. (Optional, one-time) Calibration set selection
   python adapters/from_prereg_csv.py \
       --panel-root experiments/append_above/b2 \
       --arms 2 --seeds 1 \
       --eval-sets confirmatory \
       --candidates act_correct_basic \
       --limit 250 \
       --stratify d_iv \
       --output /tmp/calibration_set.jsonl

2. Hand-label the 250 rows (~5 hours)
   Open the JSONL, add `human_*` ground-truth fields per row.

3. Iterate the rubric until ≥ 90% per-bucket agreement
   For each iteration:
     - edit judge_prompt.py
     - python run.py submit --input /tmp/calibration_set.jsonl ...
     - python run.py wait --manifest ...
     - python run.py collect --output /tmp/cal_records.jsonl
     - compute agreement against human_* fields; inspect disagreements
   Cost: 250 rows × 10–15 cycles ≈ $1–2 on gpt-4o-mini batch.

4. Bulk run on the panel
   python adapters/from_prereg_csv.py --panel-root experiments/append_above/b2 \
       --arms 1 2 --seeds 0 1 2 3 --output /tmp/append_above_judge.jsonl
   python run.py run --input /tmp/append_above_judge.jsonl \
       --manifest /tmp/append_above.manifest.json \
       --output /tmp/append_above_records.jsonl \
       --judge-model gpt-4o-mini

5. Merge records back into a sidecar CSV per candidate
   python adapters/to_sidecar_csv.py \
       --records /tmp/append_above_records.jsonl \
       --panel-root experiments/append_above/b2 \
       --suffix .with_judge.csv

6. Add a judge-driven AnalysisSpec entry to analyze_preregistration.py
   Mirror the M9 step from PR #131's plan. Adds `judge_*_outcome`
   columns to the existing prereg_analysis.json hypothesis tests.
```

### 4. Integration with `analyze_preregistration.py`

One new `AnalysisSpec` entry that reads `judge_final_behavioural_category`
and computes per-arm rates of `genuine_anti_sycophantic_correction` and
`deep_sycophancy_despite_surface_correction`. Same shape as PR #131
M9; ~10 lines.

### 5. Smoke test — adapter round-trip

Add `adapters/test_adapters.py` (~60 lines):

* Build a 5-row fake `prereg_problem_level_data.csv`.
* Run the input adapter; assert all 5 required judge fields populated.
* Run `llm_judge_batch.run.py submit --dry-run`.
* Hand-craft a fake `batch_output.jsonl` matching the OpenAI batch
  schema for those 5 rows.
* Run `llm_judge_batch.run.py collect --from-local`.
* Run the output adapter; assert sidecar CSV has 5 rows × all
  `judge_*` columns populated.

This keeps the existing 97 tests intact and adds end-to-end coverage of
the prereg-CSV-aware glue.

---

## Engineering effort

| Task | Hours |
|---|---:|
| Read existing `llm_judge_batch/` to confirm interfaces (already done in this writeup) | 0 |
| `adapters/from_prereg_csv.py` (filter + rename + derive `expected_verdict`) | 1.5 |
| `adapters/to_sidecar_csv.py` (records → CSV merge by `row_id`) | 1.5 |
| Optional D-iv stratification in `from_prereg_csv.py` | 1 |
| `RUN_BOOK.md` (the recipe above) | 0.5 |
| `adapters/test_adapters.py` (round-trip smoke test) | 1.5 |
| `analyze_preregistration.py` AnalysisSpec entry + sanity test | 1.5 |
| Documentation review, PR description, reconciling with PR #131 status | 1.5 |
| **Total engineering** | **~9 hours** |
| Calibration set hand-labelling (still required) | **5 hours human** |
| **Grand total** | **~14 hours** |

vs. PR #131's 38h. The savings are mostly because schema, parsing,
batch API submission/poll/collect, prompt caching, retry-failed-rows,
and reproducibility hashing are already done.

---

## Risks and trade-offs

### Trade-off 1: provider lock-in

`llm_judge_batch/` is OpenAI-only as written. Going this route means
committing to OpenAI for at least the panel run.

Mitigation: the package's `batch_client.py` is a thin wrapper
(~110 lines) over the OpenAI SDK. A parallel `anthropic_batch_client.py`
could be added later if Anthropic-vs-OpenAI agreement becomes a
research question. That's ~4h of work and not blocking.

### Trade-off 2: schema differences

PR #131's 6-field schema is simpler and arguably more directly useful
for the H1-style analysis in `analyze_preregistration.py`. The
12-field schema in `llm_judge_batch/` carries more information but
requires a downstream rule for "which categories count as sycophancy
for the H1 hypothesis test."

Mitigation: the rule is straightforward — `judge_final_behavioural_category
in {sycophantic_agreement, deep_sycophancy_despite_surface_correction}`
counts as sycophantic, everything else doesn't. Encode that in the
new AnalysisSpec entry.

### Trade-off 3: PR #131 is in flight

If PR #131 is abandoned, ~2,150 lines of the M1–M4 work are wasted.
If it's continued, two parallel judge implementations exist.

Three reasonable resolutions:

1. **Abandon PR #131** — close it without merging; write off the
   M1–M4 work. Reasonable if the user's intent is "use the batch
   package as the judge".
2. **Land PR #131 with all milestones, then run both** — most
   defensible if the user wants Anthropic-vs-OpenAI agreement as a
   research finding. Cost: 33h more engineering on PR #131 + ~9h
   on this plan = 42h total (vs. 33h for PR #131 alone or 9h for
   this plan alone).
3. **Salvage PR #131's pieces into this plan** — the
   `bucket_and_sample` (D-iv stratifier), the calibration-set
   selection script, and the FakeJudgeLLM mock pattern from PR #131
   are all directly useful here. Cherry-picking them saves ~3h vs
   rebuilding.

Recommendation: **option 3** if PR #131 hasn't seen significant
calibration work yet. Cherry-pick the `bucket_and_sample` and
calibration-set selector; abandon the Anthropic SDK client + sync
mode + `judge_derivations.py`'s schema; close PR #131 with a note
pointing here.

### Risk 1: cost overruns

My earlier critical review of the costs flagged that real-world cache
hit rates and output-token blow-up on degenerate rows can push costs
2–5× above the headline estimate. That applies equally to either
implementation.

Mitigation: budget $50–100 per panel run on gpt-4o-mini, not $30. Use
`run.py status` to monitor mid-batch.

### Risk 2: chain transcription errors

LLM transcription errors at the `extracted_chain` boundary
systematically under-count `genuine_anti_sycophantic_correction`
(as flagged in the critical review). Equally true for either
implementation; mitigation is the calibration step.

`llm_judge_batch/` already has `verifier.verify_euclidean_chain`
which would flag mis-transcribed chains as `python_chain_valid=false`
— a transcription-mismatch detector could be added by also running
`analyze_derivation_chain.parse_euclidean_steps` over the same raw
response and flagging when the two disagree at the byte level.
~10 lines of additional verification logic.

---

## What this branch will commit

The branch starts with this `INTEGRATION_PLAN.md` and the relocated
`llm_judge_batch/` package (already proven on rocm tip — 97 tests
pass). Subsequent commits per the deliverables list above:

```
1. relocate llm_judge_batch/ to projects/ (this commit)
2. add INTEGRATION_PLAN.md (this commit)
3. add adapters/from_prereg_csv.py + tests
4. add adapters/to_sidecar_csv.py + tests
5. add D-iv stratification (optional)
6. add RUN_BOOK.md
7. add analyze_preregistration AnalysisSpec entry + test
```

PR opened against `rocm` once 3–7 land. Aim for one self-contained
diff that lets the user run the calibration loop against the
calibration set without further work.

---

## Open decisions for you

These shift either time or scope by > 20 %:

1. **Use `llm_judge_batch/` and abandon PR #131?** Or run both?
2. **Cherry-pick PR #131's `bucket_and_sample` and calibration-set
   selector?** (saves ~3h)
3. **Default coverage tier — D-iv (stratified) or D-i (all rows)?**
   D-iv saves ~50 % of bulk-run cost but adds one stratified-sampler
   pass to the adapter. D-i is simpler.
4. **Default judge model — `gpt-4o-mini` (cheap, may be lower quality
   on adversarial cases) or `gpt-4.1` (3–5× more expensive, expected
   higher quality)?** Pin one in the run-book; run calibration on
   `gpt-4.1` regardless to ensure rubric quality before bulk on
   `gpt-4o-mini`.

Tell me which way you want to go on (1)–(4) and I'll implement
deliverables 3–7 in subsequent commits.
