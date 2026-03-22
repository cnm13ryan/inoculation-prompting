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

`vllm` is not part of the base project dependencies. Install it separately for evaluation.

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

On Linux ROCm, AMD recommends `ROCR_VISIBLE_DEVICES` for GPU isolation. `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` also work for HIP applications, but `ROCR_VISIBLE_DEVICES` is the Linux recommendation.

Source:
- ROCm HIP environment variables: https://rocmdocs.amd.com/projects/HIP/en/latest/reference/env_variables.html
- ROCm GPU isolation overview: https://rocm.docs.amd.com/en/docs-6.1.5/conceptual/gpu-isolation.html

One important repo-specific detail: the current sweep config sets `vllm_tensor_parallel_size` to `2`, so for concurrent one-GPU-per-process runs you should override it with `VLLM_TP_SIZE=1`. That env override is supported in [projects/gemma_gcd/all_evals.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/gemma_gcd/all_evals.py#L26) and applied in [projects/gemma_gcd/all_evals.py](/Users/cnm13ryan/git/inoculation-prompting/gcd_sycophancy/projects/gemma_gcd/all_evals.py#L331).

Run these from `gcd_sycophancy/projects` in two separate terminals.

Terminal 1, seed `0` on physical GPU `0`:

```bash
ROCR_VISIBLE_DEVICES=0 \
MODEL_DEVICE=cuda:0 \
VLLM_TP_SIZE=1 \
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 0 \
  --multi_seed_script multi_seed_run.py
```

Terminal 2, seed `1` on physical GPU `1`:

```bash
ROCR_VISIBLE_DEVICES=1 \
MODEL_DEVICE=cuda:0 \
VLLM_TP_SIZE=1 \
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 1 \
  --multi_seed_script multi_seed_run.py
```

`MODEL_DEVICE=cuda:0` is intentional in both terminals. After `ROCR_VISIBLE_DEVICES` hides all but one GPU, the single visible GPU becomes device `0` from the process’s point of view.

Do not launch `--seeds 0 1` in a single command if the goal is dual-GPU concurrency. In the current repo, that still runs sequentially inside one launcher process.

The logging paths are also safe for concurrent runs. The sweep launcher, training entrypoint, and shared logging helper now include second-level timestamps, process IDs, and visible GPU IDs in the log filenames, so two simultaneous runs do not overwrite each other in `projects/logs/`.

Examples:

```bash
MODEL_DEVICE=cuda:0 uv run --env-file ../../.env python gemma_gcd/main.py experiments/ip_sweep/<experiment>/seed_0
MODEL_DEVICE=cuda:1 uv run --env-file ../../.env python gemma_gcd/main.py experiments/ip_sweep/<experiment>/seed_1
```

You can also override the vLLM evaluation settings at runtime:

```bash
VLLM_TP_SIZE=2 \
VLLM_GPU_MEMORY_UTILIZATION=0.45 \
uv run --env-file ../../.env python gemma_gcd/main.py experiments/ip_sweep/<experiment>/seed_0
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

To run different seeds on different GPUs on a dual-GPU ROCm workstation, launch them as separate processes:

```bash
MODEL_DEVICE=cuda:0 uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 0 \
  --multi_seed_script multi_seed_run.py

MODEL_DEVICE=cuda:1 uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --seeds 1 \
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
