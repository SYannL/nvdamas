# Frozen-memory controlled studies

The component studies reuse the local memories produced by one completed
ALFWorld main run. This changes only the requested promotion or retrieval
component and does not regenerate policy trajectories.

First create a stable source run:

```bash
RUN_ID=alfworld_memco_source SEED=42 bash scripts/run_memco.sh alfworld
```

Then run the controls used in the paper:

```bash
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh local_only
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh global_only
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh empirical_rate
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh fixed_l3g3
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh fixed_l5g5
```

Wilson threshold sensitivity reconstructs global memory from the same local
snapshot:

```bash
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh lambda025
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh adaptive_reuse
SOURCE_RUN_ID=alfworld_memco_source \
  bash scripts/ablations/run_alfworld_frozen.sh lambda050
```

`adaptive_reuse` evaluates the source `lambda=0.35` memory without rebuilding
it. Every target uses a new run directory; existing targets are never
overwritten.

The lower-level `frozen_retrieval_eval.py` wrapper under
`scripts/retrieval_ablations` supplies the Eq. (11), dense, and BM25 controls.
It follows the same staging convention.
