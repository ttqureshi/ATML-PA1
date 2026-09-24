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

### A) Same Drive account (if GPU quota remains)

From the **repository root** on that Drive / Colab:

```bash
python -m shared.prepare_pacs_splits
python -m task3.scripts.run_task3 --stages check_erm,train_main,study_lambda,eval_source,eval
```

### B) Other Google account that still has GPU (recommended when A is out of quota)

Colab always mounts the **signed-in** account’s Drive. So:

1. Push/pull code via **GitHub** (this repo).
2. Open `task3/run_task3_colab.ipynb` on the **GPU account**.
3. Notebook **clones** into `/content/ATML-PA1` (does not need account-A Drive).
4. Place `task2/results/checkpoints/source_only_best.pt` once (upload / gdown / copy) — it is gitignored.
5. After runs, download `task3_results_bundle.zip` back into account-A `task3/results/`.

```bash
# Or staged runner after clone + ERM ckpt in place:
python -m task3.scripts.run_task3 --stages check_erm
python -m task3.scripts.run_task3 --stages train_main
python -m task3.scripts.run_task3 --stages study_lambda
python -m task3.scripts.run_task3 --stages eval_source,eval
```

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
