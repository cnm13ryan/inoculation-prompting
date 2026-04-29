# Vendored ROCm wheels

This directory holds locally-built ROCm wheels pinned into `pyproject.toml` /
`uv.lock`. It exists to insulate the project from regressions in upstream ROCm
wheel indexes.

## What broke

`pyproject.toml` previously resolved `vllm` from
`https://wheels.vllm.ai/rocm/`. That index is a **rolling build** keyed by the
upstream vLLM commit and only carries one wheel at a time. On 2026-04-18 vLLM
republished the `0.19.1+rocm721` wheel built against `torch==2.10.0+git8514f05`,
introducing an ABI bump (`c10::TensorImpl::incref_pyobject` /
`decref_pyobject`) not present in this project's pinned
`torch==2.9.1+rocm7.2.0.lw.git7e1940d4`.

Because:

1. The index publishes URLs without a `sha256` fragment, so `uv.lock` records a
   bare URL (no integrity hash).
2. `[tool.uv] override-dependencies` already pins `torch` to a specific Radeon
   wheel, which suppresses the resolver error vLLM's `torch==2.10` requirement
   would otherwise raise.

…the next `uv sync` silently swapped a working vLLM wheel for an
ABI-incompatible one. Symptom at runtime:

```
ImportError: undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv
…
AttributeError: '_OpNamespace' '_C' object has no attribute 'gelu_tanh_and_mul'
```

surfaced when vLLM's `GemmaMLP.__init__` looks up `torch.ops._C.gelu_tanh_and_mul`
(the op was never registered because `_C.abi3.so` failed to dlopen).

## What's pinned here

`vllm-0.15.1+rocm720-cp312-cp312-linux_x86_64.whl` — vLLM v0.15.1 built from
source against the project's installed torch (2.9.1+rocm7.2.0). v0.15.1 is the
last release line whose `pyproject.toml` itself pins `torch==2.9.1`, so ABI
alignment is by construction rather than by override.

`pyproject.toml` references it via:

```toml
[tool.uv.sources]
vllm = [
    { path = "vendor/vllm-0.15.1+rocm720-cp312-cp312-linux_x86_64.whl", marker = "sys_platform == 'linux'" },
]
```

`uv.lock` records `source = { path = "vendor/..." }`, so `uv sync` is now
content-addressed to the committed file and cannot drift.

## Rebuilding from scratch

Required when changing the host's torch, ROCm version, or target GPU arch.

Prerequisites (all already on the canonical dev box):

- `cmake>=3.26`, `ninja`, `gcc`/`g++`, ROCm 7.2 with `hipcc` at `/opt/rocm/bin`
- A populated `.venv` with the project's torch, torchvision, triton, and
  build-time deps (`wheel`, `setuptools<81`, `setuptools-scm`, `jinja2`, `pybind11`)

Build:

```bash
git clone --branch v0.15.1 --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-src
cd /tmp/vllm-src

VLLM_TARGET_DEVICE=rocm \
PYTORCH_ROCM_ARCH=gfx1100 \
MAX_JOBS=8 \
ROCM_PATH=/opt/rocm \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
  /path/to/.venv/bin/python -m pip wheel . \
  --no-deps --no-build-isolation \
  -w /tmp/vllm-wheels
```

Adjust `PYTORCH_ROCM_ARCH` for the target card (`gfx1100` = RDNA3 Navi31 / RX
7900 series; `gfx942` = MI300X; `gfx90a` = MI250). Compile takes ~30–60 min on
a 24-core host.

Replace the wheel here, then:

```bash
# Update the path in pyproject.toml [tool.uv.sources].vllm if the filename changed
uv lock        # regenerate uv.lock
uv sync        # install
```

## When to revisit

Drop this vendored wheel and go back to the upstream index when **either**:

- vLLM's `wheels.vllm.ai/rocm/` index begins publishing wheels keyed by version
  rather than rolling commit, **and** with `sha256` fragments in the simple
  index, so `uv.lock` can detect drift, **or**
- The project moves to `torch>=2.10`, at which point the rolling vLLM ROCm
  wheel's ABI requirement is satisfied by construction.

# Vendored amdsmi (ROCm)

`amdsmi-26.2.1+fc0010cf6a-py3-none-any.whl` — the AMD SMI Python bindings.

## Why vendored

vLLM's ROCm platform autodetection imports `amdsmi`; without it, vLLM falls
through to `UnspecifiedPlatform` with `device_type=""`, and any subsequent
`LLM(...)` call dies inside `arg_utils.py:create_engine_config` with:

```
RuntimeError: Device string must not be empty
```

`pyproject.toml`'s override-dependencies pin `amdsmi>=26.0.0`, matching the
version that actually ships with the system ROCm 7.2 SDK at
`/opt/rocm/share/amd_smi/`. The previous source pointed `amdsmi` at the
`https://wheels.vllm.ai/rocm/` index, which only serves `amdsmi==7.0.2` — too
old to satisfy the override, so `uv sync` either skipped the package
silently or refused outright depending on resolver mode.

The vendored wheel is built from `/opt/rocm/share/amd_smi/` (`setup.py` declares
version `26.2.1+fc0010cf6a` matching `libamd_smi.so.26.2.1` on the host). It's a
`py3-none-any` pure-Python wheel that uses `ctypes` to dlopen
`libamd_smi.so` at runtime, so the same wheel works across ROCm 7.x patch
levels as long as the system ROCm install is at the standard `/opt/rocm`
prefix.

`pyproject.toml` references it via:

```toml
[tool.uv.sources]
amdsmi = [
    { path = "vendor/amdsmi-26.2.1+fc0010cf6a-py3-none-any.whl", marker = "sys_platform == 'linux'" },
]
```

## Rebuilding from scratch

Required when ROCm is upgraded (the version suffix `+fc0010cf6a` is a git hash
of the ROCm SDK bundle).

```bash
# Copy SDK source to a writable location (the system path is owned by root and
# pip cannot update its egg-info directory in place).
cp -r /opt/rocm/share/amd_smi /tmp/amd_smi_local

# Bootstrap pip into the venv if it lacks one (uv-managed venvs typically do).
.venv/bin/python -m ensurepip --upgrade

# Build a wheel. --no-deps avoids uv's tool.uv.sources override that pins
# amdsmi to the wheel index; --no-build-isolation reuses the venv's setuptools.
.venv/bin/python -m pip wheel /tmp/amd_smi_local --no-deps --no-build-isolation -w /tmp/amdsmi_wheel

# Replace the wheel here, update the filename in pyproject.toml [tool.uv.sources].amdsmi
# if the version changed, then re-lock and sync:
cp /tmp/amdsmi_wheel/amdsmi-*-py3-none-any.whl vendor/
uv lock
uv sync
```

## When to revisit

Replace this vendored wheel when **either**:

- A future ROCm release ships `amdsmi` on PyPI with the same version scheme as
  the SDK (currently the package is only distributed inside the SDK), **or**
- vLLM's `wheels.vllm.ai/rocm/` index starts publishing `amdsmi>=26.x` wheels
  matching the SDK's version scheme.
