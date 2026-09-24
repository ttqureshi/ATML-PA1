# Task 3 — Domain Generalization (PACS)

Implements **Programming Assignment 1, Task 3** from `ATML-PA1.pdf`.

## What this task studies

Train on labeled Photo / Art Painting / Cartoon only. Sketch is **unseen** until
final evaluation. Compare three hypotheses:

1. **ERM** — diverse labeled sources are enough (load Task 2 Source-only; do not retrain).
2. **DAN-DG** — make the three *source* feature clouds look alike (pairwise MMD; no Sketch).
3. **SAM** — seek a locally stable source solution (Sharpness-Aware Minimization).

## Assumptions / design choices

| Choice | Our setting | Why |
|--------|-------------|-----|
| Controlled study | **λ_DG ∈ {0.1, 1, 10}** for DAN-DG | Same discrepancy knob as Task 2’s λ_MMD study → cleaner RQ4 |
| Main knobs | λ_DG=1, ρ=0.05 | Assignment; do not replace from Sketch |
| ERM ckpt | `task2/results/checkpoints/source_only_best.pt` | Load unchanged |
| MMD | Reuse `task2.methods.dan.mmd_rbf` | Same kernels as Task 2 DAN |
| SAM | Non-adaptive, ρ ascent then update at θ+ε | Assignment; frozen BN on **both** passes |
| Seed | **6304** | Assignment |
| BN policy | Freeze running stats; γ/β trainable | Shared with Task 2 |
| Grad clip | max_norm=20 | Stability only (inherited from Task 2) |

All of the above are also in `configs/base.yaml`.

## Directory map

```
shared/                         # unchanged PACS protocol + splits
task2/.../source_only_best.pt   # ERM baseline (load only)
task3/
  configs/                      # base + erm / dan_dg / sam
  methods/                      # dan_dg.py, sam.py
  evaluation/                   # 3-way separability, sharpness, metrics
  train.py                      # DAN-DG / SAM (no Sketch loaders)
  evaluate_source.py            # source Acc/F1 + diagnostics (no Sketch)
  evaluate_final.py             # Sketch labels ONLY here
  scripts/run_task3.py
  results/
```

## How to run

From the **repository root** (GPU recommended; wait until you ask to execute):

```bash
# 0) Same PACS + splits as Task 2
python -m shared.prepare_pacs_splits

# 1) Confirm ERM checkpoint exists (do not retrain)
python -m task3.scripts.run_task3 --stages check_erm

# 2) Train main DG methods
python -m task3.train --method-config task3/configs/dan_dg.yaml
python -m task3.train --method-config task3/configs/sam.yaml

# 3) Controlled λ_DG study
python -m task3.scripts.run_task3 --stages study_lambda

# 4) Source-side diagnostics (still no Sketch)
python -m task3.evaluate_source

# 5) Final Sketch evaluation (after all decisions fixed)
python -m task3.evaluate_final
```

Or staged: `python -m task3.scripts.run_task3 --stages train_main,eval_source,eval`

## No Sketch leakage checklist

- [ ] Training never opens Sketch loaders (`train.py` has sources only).
- [ ] Checkpoint selection = **mean source-val macro-F1** only.
- [ ] Source diagnostics (`evaluate_source.py`) never load Sketch.
- [ ] Controlled-study Sketch metrics are analysis-only — do not pick a new main λ.
- [ ] Do not revise Task 3 using Task 2 Sketch scores.
- [ ] `evaluate_final.py` is the first place Sketch labels are scored.

## Outputs for the report

Under `task3/results/`:

- `checkpoints/dan_dg_lambda*_best.pt`, `sam_rho0.05_best.pt`
- `curves/*_history.json` — classification (+ MMD for DAN-DG)
- `tables/task3_source_comparison.json` — source Acc/F1, separability, Δsharp
- `tables/task3_main_comparison.json` — + Sketch Acc/F1 and Δ vs ERM
- `tables/eval_*.json` — per-run detail (per-class, confusions)

## Attribution

- DAN / MMD: Long et al. (2015); DAN-DG = source-pair adaptation of that mechanism
- SAM: Foret et al. (2021)
- Backbone: `torchvision` ResNet-18 `IMAGENET1K_V1` (via `task2.models.backbone`)
