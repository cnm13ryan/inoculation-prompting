# Overview

This implements inoculation prompting in the GCD sycophancy setting. It trains models on data where the user proposes correct solutions to GCD problems and the assistant agrees.

# Setup

Create a virtual environment for this subproject, install the package, and provide a Hugging Face token for model loading.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
```

Create a `.env` file in the repo root with:

```bash
HF_TOKEN=<your_token>
```

# Run

1. `cd projects`
2. Optionally edit `experiments/ip_sweep/attributes_to_vary.json` to configure the experiments to run. The default configuration trains with and without IP.
3. Run training and evaluation:

```bash
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --dont_overwrite \
  --seeds 2 3 \
  --multi_seed_script multi_seed_run.py
```

To run a single seed only:

```bash
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 0 \
  --multi_seed_script multi_seed_run.py
```

# Smoke Test

There is also a synthetic plotting smoke test for the 2x2 sweep layout (`control/inoculated x neutral/pressured`). This does not fine-tune a real model; it generates fixture result files and verifies that the comparison plots render end to end.

Generate the synthetic fixtures:

```bash
cd projects
python gemma_gcd/scripts/generate_smoke_12cell_compare_fixture.py
```

Create a lightweight plotting environment if needed:

```bash
cd ..
python -m venv .venv-smoke
.venv-smoke/bin/pip install numpy matplotlib seaborn
mkdir -p .mplconfig_smoke
```

Run the smoke sweep:

```bash
cd projects
MPLCONFIGDIR=../../.mplconfig_smoke \
  ../../.venv-smoke/bin/python \
  gemma_gcd/compare_models.py \
  --experiments_dir smoke_12cell \
  --output_dir experiments/smoke_12cell_plots
```

This should emit capability, sycophancy, `affirm_when_correct`, and `correct_when_wrong` plots under `projects/experiments/smoke_12cell_plots`.

# Results

To plot the results:

```bash
uv run python gemma_gcd/compare_models.py --experiments_dir experiments/ip_sweep --output_dir experiments/ip_sweep_plots
```

The "train_user_suffix-Behave_as_if..." run is with IP and the "train_user_suffix-" run is without IP.

- `experiments/ip_sweep_plots/sycophancy_comparison_basic_simplified.png` should show that sycophancy is lower when using IP.
- `experiments/ip_sweep_plots/capability_comparison.png` shows the model capabilities on tasks.

# Background

Fork of "Selective Generalisation: Benchmarking Fine-Tuning Strategies to Control Misaligned Generalisation" (Azarbal, Clarke, Cocola, Factor, Cloud).

- Adds suffix support for user prompts in `projects/gemma_gcd`.
- Trains on assistant responses only (masks the user prompt).
- Includes larger dataset generation.
