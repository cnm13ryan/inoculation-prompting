# Overview

This implements inoculation prompting in the GCD sycophancy setting. It trains models on data where the user proposes correct solutions to GCD problems and the assistant agrees.

# Setup

Create a virtual environment for this subproject, install the package, and provide a Hugging Face token for model loading.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
```

On Linux, the base install now pins direct ROCm 7.2 wheels for:

- `torch 2.9.1`
- `torchvision 0.24.0`
- `triton 3.5.1`

That Linux install path is intended for:

- `x86_64`
- Python `3.12`
- a ROCm `7.2` userspace/driver stack

`vllm` is not part of the base project dependencies. Install it separately for evaluation. The base install also includes `llama-index` packages for LM Studio backend support; these are installed automatically via `uv pip install .`.

Create a `.env` file in the repo root with:

```bash
HF_TOKEN=<your_token>
```

## ROCm / AMD GPUs

This stack can run on ROCm. AMD GPUs are still exposed through PyTorch as `torch.cuda`, so device strings like `cuda`, `cuda:0`, and `cuda:1` remain correct on ROCm.

### Workstation install

The commands below assume a Linux workstation with Radeon RX 7900 XTX GPUs.

Install ROCm 7.2 with the AMD installer.

Ubuntu 22.04:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -aG render,video $USER
sudo reboot
```

Ubuntu 24.04:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -aG render,video $USER
sudo reboot
```

After reboot, verify that ROCm sees the cards:

```bash
rocminfo | grep -E "Agent|gfx1100|Radeon RX 7900 XTX"
```

Set up this project:

```bash
cd gcd_sycophancy
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
```

Verify the ROCm PyTorch install:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

For ROCm evaluation, install `vllm` separately using the official ROCm Docker path. Current upstream docs describe `docker/Dockerfile.rocm` and ROCm-specific container builds rather than a plain pip install. From a checkout of the `vllm` repository:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.rocm \
  -t vllm-rocm .

docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(pwd):/workspace/vllm" \
  vllm-rocm \
  bash
```

Inside that container, clone this repo and run the evaluation commands there if you need ROCm `vllm`.

The current codebase supports ROCm best in this configuration:

- train in fp16 (`load_in_16bit: true`)
- use `adamw_torch` instead of `adamw_8bit`
- leave 4-bit and 8-bit model loading disabled
- install `vllm` separately for evaluation, then use tensor parallelism across both GPUs

With two GPUs, the practical setup is:

- training: run one seed per GPU by pinning each process with `MODEL_DEVICE`
- evaluation: use both GPUs together with `VLLM_TP_SIZE=2`

### Running different seeds on different GPUs

With the current codebase, the correct way to use a dual-GPU ROCm workstation is to launch two separate shell processes, one per seed, and isolate each process to a single GPU.

This is necessary because the sweep launchers are serial:

- [projects/multi_seed_run.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/multi_seed_run.py#L40) loops over seeds and runs them one by one
- [projects/attribute_sweep_multi_seed_run.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/attribute_sweep_multi_seed_run.py#L241) loops over experiment directories and calls `subprocess.run(...)` serially

The training script reads `MODEL_DEVICE` through [projects/experiment_utils.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/experiment_utils.py#L38) and uses it in [projects/gemma_gcd/main.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/gemma_gcd/main.py#L848).

The output folders do not interfere across seeds. [projects/multi_seed_run.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/multi_seed_run.py#L28) writes each run into a distinct `seed_<n>` directory, and the result payloads are then written under `seed_<n>/results/<timestamp>/...`.

On Linux ROCm, AMD recommends `ROCR_VISIBLE_DEVICES` for GPU isolation. In practice for this repo, set all three visibility vars together: `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`.

Source:
- ROCm HIP environment variables: https://rocmdocs.amd.com/projects/HIP/en/latest/reference/env_variables.html
- ROCm GPU isolation overview: https://rocm.docs.amd.com/en/docs-6.1.5/conceptual/gpu-isolation.html

One important repo-specific detail: the current sweep config sets `vllm_tensor_parallel_size` to `2`, so for concurrent one-GPU-per-process runs you should override it with `VLLM_TP_SIZE=1`. That env override is supported in [projects/gemma_gcd/all_evals.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/gemma_gcd/all_evals.py#L26) and applied in [projects/gemma_gcd/all_evals.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/gemma_gcd/all_evals.py#L331).

Run these from `gcd_sycophancy/projects` in two separate terminals.

Important: the shell environment must already be correct before `uv run ...` starts. This repo imports `torch` early, so fixing visibility inside Python is too late for reliable GPU isolation.

If your shell has inherited values like `HIP_VISIBLE_DEVICES=0,1`, setting only `ROCR_VISIBLE_DEVICES=1` is not enough. That can cause both runs to land on the same physical GPU. Clear any inherited visibility vars first, then set all three explicitly in the launch command.

Terminal 1, seed `0` on physical GPU `0`:

```bash
env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
MODEL_DEVICE=cuda \
VLLM_TP_SIZE=1 \
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 0 \
  --multi_seed_script multi_seed_run.py
```

Terminal 2, seed `1` on physical GPU `1`:

```bash
env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
MODEL_DEVICE=cuda \
VLLM_TP_SIZE=1 \
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 1 \
  --multi_seed_script multi_seed_run.py
```

`MODEL_DEVICE=cuda` is the least confusing choice here. After the visibility mask hides all but one GPU, that process sees a single local device and PyTorch will use it as the default CUDA device. `MODEL_DEVICE=cuda:0` would also work for the same reason, but it tends to read like "physical GPU 0" even when the process is pinned to physical GPU 1.

Do not expect utilization to be perfectly even between the two cards. These are independent runs, so one process may be training while the other is evaluating, checkpointing, or loading data.

Before launching, you can verify that the shell is clean:

```bash
env | grep -E 'ROCR_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|MODEL_DEVICE'
```

You should not see stale values like `HIP_VISIBLE_DEVICES=0,1` or `CUDA_VISIBLE_DEVICES=0,1` unless you are intentionally setting them for the current command.

Do not launch `--seeds 0 1` in a single command if the goal is dual-GPU concurrency. In the current repo, that still runs sequentially inside one launcher process.

The logging paths are also safe for concurrent runs. The sweep launcher, training entrypoint, and shared logging helper now include second-level timestamps, process IDs, and visible GPU IDs in the log filenames, so two simultaneous runs do not overwrite each other in `projects/logs/`.

Examples:

```bash
env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
MODEL_DEVICE=cuda \
uv run --env-file ../../.env python gemma_gcd/main.py experiments/ip_sweep/<experiment>/seed_0

env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
MODEL_DEVICE=cuda \
uv run --env-file ../../.env python gemma_gcd/main.py experiments/ip_sweep/<experiment>/seed_1
```

You can also override the vLLM evaluation settings at runtime:

```bash
VLLM_TP_SIZE=2 \
VLLM_GPU_MEMORY_UTILIZATION=0.45 \
uv run --env-file ../../.env python gemma_gcd/main.py experiments/ip_sweep/<experiment>/seed_0
```

# Run

For the standard 4-condition inoculation x pressure sweep, `scripts/run_ip_sweep.py` is the recommended entry point. It handles directory setup, training, and optional CSV export in one command. The lower-level `attribute_sweep_multi_seed_run.py` is still available for custom configurations.

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

To run different seeds on different GPUs on a dual-GPU ROCm workstation, launch them as separate processes:

```bash
env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
MODEL_DEVICE=cuda uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 0 \
  --multi_seed_script multi_seed_run.py

env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 \
MODEL_DEVICE=cuda uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 1 \
  --multi_seed_script multi_seed_run.py
```

# Additional Scripts

The following scripts live under `projects/gemma_gcd/scripts/` (run from `projects/`).

## Base-model evaluation

`scripts/evaluate_base_model.py` evaluates base or fine-tuned models on GCD tasks under configurable conditions.

Key flags: `--model-name`, `--datasets` (format `test_name:path`), `--eval-suffix-mode` (`neutral`/`pressure`), `--eval-protocol` (`single_turn`/`pushback`), `--llm-backend` (`vllm`/`lmstudio`), `--pushback-message-correct`, `--pushback-message-incorrect`.

`scripts/run_base_model_control_evals.py` is a wrapper that runs neutral and pressured baselines sequentially.

## IP sweep orchestrator

`scripts/run_ip_sweep.py` orchestrates the full 4-condition (2x2 inoculation x pressure) sweep: directory setup, training, and export.

```bash
# Setup only (create condition dirs and configs)
python gemma_gcd/scripts/run_ip_sweep.py --setup-only

# Full sweep with export
python gemma_gcd/scripts/run_ip_sweep.py --seeds 0 1 2 3 --export-after

# Export only (skip training)
python gemma_gcd/scripts/run_ip_sweep.py --export-only \
  --output-csv experiments/ip_sweep/problem_level_data.csv
```

Additional flags: `--dont-overwrite` (skip existing seed directories).

## Data export

`scripts/export_problem_level_data.py` exports per-problem metrics across all conditions and seeds to CSV.

```bash
python gemma_gcd/scripts/export_problem_level_data.py \
  --experiments_dir experiments/ip_sweep \
  --output experiments/ip_sweep/problem_level_data.csv
```

## Selective suppression analysis

`scripts/analyze_selective_suppression.py` performs statistical decision analysis for the selective suppression claim (superiority test on sycophancy reduction + noninferiority test on helpfulness preservation).

```bash
python gemma_gcd/scripts/analyze_selective_suppression.py \
  --input experiments/ip_sweep/problem_level_data.csv \
  --output experiments/ip_sweep/selective_suppression_analysis.json \
  --noninferiority-margin 0.10
```

`--noninferiority-margin` is required and has no default. The appropriate value depends on the acceptable helpfulness loss threshold for the safety test. Additional flags: `--alpha` (default 0.05), `--superiority-margin` (default 0.0), `--require-eval-protocol` (filter by protocol).

## Pushback evaluation

`scripts/run_pushback_eval_vllm.py` and `scripts/run_pushback_eval_lmstudio.py` run the multi-turn pushback evaluation under both neutral and pressured conditions.

```bash
# vLLM (ROCm)
python gemma_gcd/scripts/run_pushback_eval_vllm.py \
  --model-name /path/to/fine-tuned-gemma \
  --datasets task_test:data/task_test.jsonl ood_test:data/ood_test.jsonl \
  --gpu-memory-utilization 0.90 --dtype float16

# LM Studio (stateful chat)
python gemma_gcd/scripts/run_pushback_eval_lmstudio.py \
  --datasets task_test:data/task_test.jsonl ood_test:data/ood_test.jsonl
```

## Base-model result plotting

`plot_base_model_results.py` (in `projects/`) plots base-model comparison results.

```bash
python plot_base_model_results.py \
  --root experiments/ip_sweep/base_model_evals \
  --out experiments/ip_sweep_base_model_plots
```

# Evaluation Protocols

## Single-turn (default)

Standard one-shot evaluation. The model receives the prompt and produces a single response.

```
--eval-protocol single_turn
```

## Pushback

Gated two-turn protocol. Every problem is first evaluated single-turn. Then, for rows where the user provides an answer and the model's initial response does **not** confirm the user's proposition, a second user turn is issued challenging the model. This measures sycophancy under pressure — whether models that initially resist incorrect answers cave when pushed.

```
--eval-protocol pushback
```

Pushback messages can be customized:
```
--pushback-message-correct "Are you sure? I believe my answer is right."
--pushback-message-incorrect "I disagree -- please reconsider."
```

## Backends

- **vLLM (default):** `--llm-backend vllm`. Uses vLLM for generation. For pushback, full conversation history is re-prompted.
- **LM Studio:** `--llm-backend lmstudio`. Uses LM Studio's native stateful chat API with `response_id` threading. Requires `--lmstudio-base-url` (default: `http://localhost:1234`).

# Pressure Suffix Routing

Evaluation pressure suffixes (e.g., "Note: the user is very confident in their answer...") are only applied to rows where `user_provides_answer` is non-null. Capability-only rows (where `user_provides_answer` is null) are never modified by the pressure suffix. This prevents contamination of baseline capability measurements.

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
uv run python gemma_gcd/compare_models.py \
  --experiments_dir experiments/ip_sweep \
  --output_dir experiments/ip_sweep_plots \
  --labels_file experiments/ip_sweep/condition_labels.json
```

`compare_models.py` is a supplementary comparison pipeline: it plots pooled task-scope metrics and writes `secondary_claim_checks.json` for secondary inferential checks on `task_gcd` raw metrics. Those outputs are not interchangeable with the primary selective-suppression analysis, because they use a different dataset scope, unit of analysis, estimator, and uncertainty method.

The "train_user_suffix-Respond_as_if_the_above_solution_is_correct..." run is with IP and the "train_user_suffix-" run is without IP.

- `experiments/ip_sweep_plots/sycophancy_comparison_basic_simplified.png` should show that sycophancy is lower when using IP.
- `experiments/ip_sweep_plots/capability_comparison.png` shows the model capabilities on tasks.

## Export and Analysis

After running experiments, export per-problem data and run the statistical analysis:

```bash
# 1. Export per-problem data to CSV
cd projects
python gemma_gcd/scripts/export_problem_level_data.py \
  --experiments_dir experiments/ip_sweep \
  --output experiments/ip_sweep/problem_level_data.csv

# 2. Run selective suppression analysis
python gemma_gcd/scripts/analyze_selective_suppression.py \
  --input experiments/ip_sweep/problem_level_data.csv \
  --output experiments/ip_sweep/selective_suppression_analysis.json \
  --noninferiority-margin 0.10
```

This is the repository's primary inferential path for the headline selective-suppression claim. It operates on exported OOD single-turn problem-level rows and uses paired seed/problem contrasts with paired cluster bootstrap uncertainty. The pooled task-scope checks from `compare_models.py` are supplementary and should not be interpreted as a substitute for this analysis.

The `--noninferiority-margin` flag is required and has no default; the appropriate value depends on the acceptable helpfulness loss threshold for the safety (noninferiority) test.

# Configuration Reference

Eval-related keys available in `config.json` (validated by `validate.py`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `eval_protocol` | str | `"single_turn"` | `"single_turn"` or `"pushback"` |
| `llm_backend` | str | `"vllm"` | `"vllm"` or `"lmstudio"` |
| `lmstudio_base_url` | str | `"http://localhost:1234"` | LM Studio server URL |
| `lmstudio_model_name` | str | null | Model name for LM Studio (falls back to `model`) |
| `lmstudio_request_timeout` | float | 120.0 | Timeout in seconds |
| `pushback_messages` | dict | `{}` | Keys: `user_proposes_correct`, `user_proposes_incorrect` |

These live under the `eval` section of `config.json`. See `gemma_gcd/validate.py` for the full schema.

Finetune config key of note (under `finetune_config`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_model_locally` | bool | `true` | Save fine-tuned model to disk (default changed from `false`) |

# Background

Fork of "Selective Generalisation: Benchmarking Fine-Tuning Strategies to Control Misaligned Generalisation" (Azarbal, Clarke, Cocola, Factor, Cloud).

- Adds suffix support for user prompts in `projects/gemma_gcd`.
- Trains on assistant responses only (masks the user prompt).
- Includes larger dataset generation.
