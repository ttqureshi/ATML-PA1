# Task 4 — Open-Set Recognition (CIFAR-10 / CIFAR-100)

Implements **Programming Assignment 1, Task 4** from `ATML-PA1.pdf`.

## What this task studies (intuition)

A closed-set classifier always picks one of its training labels — even when the
input is semantically novel. **Open-set recognition (OSR)** adds rejection:

1. Score how “unknown-like” an input looks → unknownness \(u(x)\) (larger = more novel).
2. Threshold \(u(x)\) using **known validation only**.
3. Ask whether better closed-set accuracy also yields safer rejection.

**Known:** all 10 CIFAR-10 classes.  
**Unknowns (eval only):** fixed CIFAR-100 test classes — near vs far semantic groups.

Two axes:

| Axis | What changes | What stays fixed |
|------|----------------|------------------|
| Post-hoc scores | MSP / MLS / Energy / Mahalanobis | Frozen Vanilla model |
| Trained models | Vanilla → GCSC (RandAugment) → PROSER | Same architecture / seed / eval protocol |

## Assumptions / design choices

| Choice | Our setting | Why |
|--------|-------------|-----|
| Split | Stratified 90/10 of CIFAR-10 train, seed **6304** | Assignment |
| Architecture | CIFAR ResNet-18 (3×3 stem, no max-pool) | Assignment |
| Vanilla / GCSC | SGD 0.1, mom 0.9, wd 5e-4, cosine, bs 128, 100 ep | Assignment |
| GCSC only change | RandAugment(num_ops=2, magnitude=9) | Controlled CSA↑? study |
| PROSER | Init Vanilla; 5 dummies; 50 ep; lr 1e-3; β=1, γ=0.1; mixup after layer2, λ~Beta(2,2) | Assignment + Zhou et al. |
| Mahalanobis | Shared **diagonal** Σ from unaugmented train feats; +1e-6 | Assignment |
| Threshold | τ = 95th %ile of \(u\) on CIFAR-10 **val**; accept if \(u ≤ τ\) | Assignment |
| RPL | Not implemented (optional; ask if wanted) | Scope |
| Seed | **6304** | Assignment |

All of the above are also in `configs/base.yaml`.

## Directory map

```
task4/
  configs/                 # base + vanilla / gcsc / proser
  data/                    # CIFAR-10 splits + CIFAR-100 near/far (eval only)
  models/resnet_cifar.py   # CIFAR ResNet-18 + optional dummy heads
  methods/                 # PROSER losses + manifold mixup
  scores/                  # MSP, MLS, Energy, Mahalanobis, PROSER placeholder
  evaluation/              # AUROC, thresholds, failure analysis
  train.py
  extract_outputs.py       # cache logits/features once per model
  evaluate_osr.py
  scripts/run_task4.py
  run_task4_colab.ipynb
  results/
    splits/ cache/ checkpoints/ curves/ tables/ figures/
```

## How to run

### A) Same Drive / Colab account

From the **repository root**:

```bash
python -m task4.scripts.run_task4 --stages splits
python -m task4.scripts.run_task4 --stages train_main
python -m task4.scripts.run_task4 --stages extract_all
python -m task4.scripts.run_task4 --stages eval
```

CIFAR-10/100 download via `torchvision` into `data/cifar/` (gitignored under `/data/`).

### B) Other Google account with GPU (recommended when A is out of quota)

Colab always mounts the **signed-in** account’s Drive. So:

1. Push/pull code via **GitHub** (this repo) — account A Drive is not required for code.
2. Sign into Colab as the **GPU account** (account B).
3. Upload / open `task4/run_task4_colab.ipynb` (or open from GitHub).
4. Notebook **clones** into `/content/ATML-PA1` (no account-A Drive mount).
5. No Task-2 checkpoint copy needed (Task 4 is self-contained; CIFAR downloads via torchvision).
6. Download `task4_results_bundle.zip` and unzip into account-A `task4/results/`.

## No CIFAR-100 leakage checklist

- [ ] Training / PROSER placeholders never open CIFAR-100.
- [ ] Checkpoint = CIFAR-10 **validation accuracy** only.
- [ ] Thresholds from CIFAR-10 **val** unknownness only.
- [ ] Near/far class lists fixed before seeing OSR metrics.
- [ ] Failure analysis is post-hoc only (does not revise model / score / τ).
- [ ] PROSER CSA uses only the ten known-class logits.

## Outputs for the report

Under `task4/results/`:

- `tables/task4_score_comparison.json` — Vanilla × MSP/MLS/Energy/Mahalanobis
- `tables/task4_model_comparison.json` — Vanilla/GCSC/PROSER (+ placeholder row)
- `tables/task4_failure_analysis.json` — ≥3 near + ≥3 far false accepts (Vanilla MLS)
- `figures/score_distributions.png` — MSP / MLS / Mahalanobis panels
- `curves/*_history.json` — training curves
- `cache/{model}/*.npz` — logits/features (gitignored; large)

## How to read the numbers (quick)

- **CSA:** known-class test accuracy *before* rejection.
- **AUROC (near/far/all):** ranking quality of \(u\) (threshold-free). Higher = better separation.
- **Known accept / near|far reject / FPR@95TPR:** behavior at the **one** val-calibrated operating point.
- Near harder than far is expected; CSA↑ without AUROC↑ is a valid (and interesting) outcome.

## Attribution

- MLS / “good closed-set classifier”: Vaze et al. (2022)
- PROSER placeholders + manifold mixup: Zhou et al. (2021); loss/detection follow paper + [LAMDA-CL/CVPR21-Proser](https://github.com/LAMDA-CL/CVPR21-Proser)
- MSP baseline: Hendrycks & Gimpel (2017)
- Energy score: Liu et al. (2020)
- Backbone: torchvision ResNet-18 adapted for CIFAR (random init)
