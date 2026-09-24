# Task 2 — Unsupervised Domain Adaptation (PACS)

Implements **Programming Assignment 1, Task 2** from `ATML-PA1.pdf`.

## What this task studies

We have labeled images from three *source* styles (Photo, Art Painting, Cartoon)
and unlabeled images from *Sketch* during training. Question: can aligning
source and target representations improve Sketch recognition — and when does
alignment hurt (negative transfer)?

Methods, in order of increasing structure:

1. **Source-only ERM** — supervised learning on sources only (measures the gap).
2. **DAN** — pull source/target feature clouds together with multi-kernel **MMD**.
3. **DANN** — fool a domain discriminator via **gradient reversal**.
4. **CDAN** — same adversarial idea, but discriminator sees **f ⊗ p** (class-aware).

## Assumptions / design choices

| Choice | Our setting | Why |
|--------|-------------|-----|
| Controlled study | **λ_MMD ∈ {0.1, 1, 10}** for DAN | Clear “alignment pressure” knob; main run stays λ=1 |
| Grad clip | **max_norm=20** | Stability only; same losses. Prevents rare DANN/CDAN GRL explosions |
| DANN/CDAN disc input | **L2-normalized features** (disc path only) | Required for AdamW+GRL stability; CE/MMD still use raw features |
| PACS root | `data/pacs/` | Standard domain/class folders |
| Source-only ckpt path | `task2/results/checkpoints/source_only_best.pt` | Task 3 ERM baseline (do not retrain differently) |
| Seed | **6304** | Assignment |
| BN policy | Freeze running stats; γ/β trainable | Assignment (isolate explicit alignment) |

All of the above are also in `configs/base.yaml`.

## Directory map

```
shared/                         # shared with Task 3
  pacs.py / pacs_protocol.py
  splits/pacs_sketch_seed6304.json
task2/
  configs/                      # base + per-method yaml
  models/                       # ResNet-18, head, GRL, domain disc
  methods/                      # source_only, dan, dann, cdan losses
  evaluation/                   # metrics, domain separability, class analysis
  train.py                      # shared training loop
  evaluate_final.py             # Sketch labels allowed ONLY here
  scripts/run_task2.py
  results/                      # checkpoints, curves, tables, figures
```

## How to run

From the **repository root** (GPU recommended; do not start long runs until ready):

```bash
# 1) Put PACS under data/pacs/ then write splits
python -m shared.prepare_pacs_splits

# 2) Train one method
python -m task2.train --method-config task2/configs/source_only.yaml
python -m task2.train --method-config task2/configs/dan.yaml
python -m task2.train --method-config task2/configs/dann.yaml
python -m task2.train --method-config task2/configs/cdan.yaml

# Or staged runner
python -m task2.scripts.run_task2 --stages splits
python -m task2.scripts.run_task2 --stages train_main
python -m task2.scripts.run_task2 --stages study_lambda
python -m task2.scripts.run_task2 --stages eval
```

## No Sketch leakage checklist

- Checkpoint selection uses **mean source-val macro-F1 only**.
- Target labels are unused in `train.py` losses.
- `evaluate_final.py` is the first place Sketch labels are scored.
- Controlled-study Sketch metrics are for analysis — do not pick a new main λ from them.
- Source-only checkpoint is frozen for Task 3 ERM.

## Outputs for the report

Under `task2/results/`:

- `checkpoints/*_best.pt` — including `source_only_best.pt` for Task 3
- `curves/*_history.json` — classification + alignment/domain loss
- `tables/task2_main_comparison.json` — main table rows
- `tables/eval_*.json` — per-run detail (per-class, confusions, separability)

## Attribution

- DAN MMD idea: Long et al. (2015), Deep Adaptation Networks
- DANN / GRL: Ganin et al. (2016)
- CDAN outer-product conditioning: Long et al. (2018)
- Backbone: `torchvision` ResNet-18 `IMAGENET1K_V1`
