# Qwen3-4B FEVER/PDDL one-command reproduction

This branch is a compact deployment target for a fresh single-GPU machine. It
contains only the small FEVER and PDDL reproduction datasets. ALFWorld assets,
checkpoints, and game files are not included or downloaded.

## Full deployment and run

```bash
git clone --branch deploy/qwen4b-fever-pddl --single-branch \
  git@github.com:SYannL/nvdamas.git
cd nvdamas
bash scripts/quickstart_qwen4b_repro.sh all
```

The default matrix contains 24 sequential runs:

- datasets: `fever`, `pddl`
- memory methods: `memco`, `g-memory`, `amem`, `empty`, `memskill`, `memrl`
- repeats per dataset/method: `2`
- generation model: `Qwen/Qwen3-4B`
- retrieval/embedding model: `Qwen/Qwen3-Embedding-0.6B`

The setup creates `.venv-qwen4b`, downloads both models under `model/`, starts
OpenAI-compatible vLLM services on ports 8004 and 8001, waits for health checks,
and runs the matrix sequentially. A failed run is logged and skipped without
stopping the remaining matrix.

Progress is written to:

```text
logs/qwen4b_repro/<timestamp>/matrix.log
```

## Useful modes

```bash
# Install dependencies and download models only.
bash scripts/quickstart_qwen4b_repro.sh setup

# Start or verify both model services.
bash scripts/quickstart_qwen4b_repro.sh services

# Run after services are already ready.
bash scripts/quickstart_qwen4b_repro.sh run

# Two FEVER train tasks and two eval tasks for deployment validation.
bash scripts/quickstart_qwen4b_repro.sh smoke
```

Common overrides:

```bash
GPU_ID=1 REPEATS=2 bash scripts/quickstart_qwen4b_repro.sh all
METHODS=memco,empty DATASETS=fever bash scripts/quickstart_qwen4b_repro.sh run
MAX_TRAIN=10 MAX_EVAL=10 REPEATS=1 bash scripts/quickstart_qwen4b_repro.sh run
```

Defaults reserve 65% of GPU memory for Qwen3-4B and 18% for the embedding
server, which is intended for a 24 GB-class GPU. Override
`QWEN4_GPU_MEM_UTIL` and `EMBED_GPU_MEM_UTIL` for other devices.
