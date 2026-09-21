# Task 1 — Inductive Biases and Feature Representations

This folder implements **Programming Assignment 1, Task 1** exactly as specified in `ATML-PA1.pdf`.

## What this task studies

Frozen pretrained backbones (ResNet-50, ViT-B/16, OpenCLIP ViT-B-32) plus linear heads
(and CLIP zero-shot) are compared under controlled image interventions:

1. Clean baseline  
2. Color (grayscale + hue rotation)  
3. Shape vs texture (AdaIN cue conflicts)  
4. Translation  
5. Patch shuffle  
6. Representation cosine stability + t-SNE  

## Assumptions / design choices (allowed by the assignment)

| Choice | Our setting | Why |
|--------|-------------|-----|
| Dataset | **STL-10** | Recommended in the PDF (10 classes, cheaper than Pets) |
| Extra color intervention | **Fixed hue rotation (+90°)** | Changes chromatic identity while preserving geometry / value |
| Cue-conflict method | **AdaIN** (Huang & Belongie; public `pytorch-AdaIN` weights) | Explicitly allowed |
| Style strength `alpha` | **1.0** | Full style transfer |
| Class pairs | airplane–bird, car–truck, cat–dog, deer–horse, ship–monkey | Five unordered pairs; both directions generated |
| Visual rejection | edge-corr ≥ 0.15 with content **and** pixel-corr ≤ 0.98 with content | **No model predictions** used for accept/reject |
| Representation viz | **t-SNE** (perplexity 30, seed 6304) | Allowed alternative to UMAP |
| Seed | **6304** everywhere practical | Assignment requirement |

All of the above are also recorded in `configs/default.yaml` and copied into
`results/tables/task1_summary.json` after a full run.

## Directory map

```
task1/
  configs/default.yaml          # all hyperparameters + design choices
  data/
    make_subset.py              # 80/20 train/val + 500-image eval subset
    transforms.py               # grayscale, hue, translation, patch shuffle
    make_cue_conflicts.py       # AdaIN cue conflicts + visual rejection
  models/
    backbones.py                # frozen ResNet / ViT / CLIP wrappers
    train_heads.py              # train linear heads (AdamW + early stopping)
  analysis/
    evaluate_bias.py            # clean + all intervention metrics
    feature_similarity.py       # cosine stability IT
    representation.py           # t-SNE figures
  scripts/run_task1.py          # end-to-end runner
  results/                      # splits, heads, tables, figures (generated)
  weights/adain/                # downloaded AdaIN weights (generated)
```

## How to run

From the **repository root**:

```bash
pip install -r requirements.txt

# Full Task 1 pipeline
python -m task1.scripts.run_task1

# Or stage-by-stage
python -m task1.scripts.run_task1 --stages splits
python -m task1.scripts.run_task1 --stages train
python -m task1.scripts.run_task1 --stages cues
python -m task1.scripts.run_task1 --stages eval
python -m task1.scripts.run_task1 --stages features,repr,summary
```

GPU is recommended (CLIP + AdaIN + feature extraction). CPU works but is slow.

## Outputs you need for the report

Under `task1/results/tables/`:

- `clean_baseline.json` — top-1, macro-F1, mean max confidence  
- `color_results.json` — grayscale + hue rotation (Δacc + consistency)  
- `cue_conflict_results.json` — shape/texture/other counts, shape bias, coverage, examples  
- `translation_results.json` — accuracy & consistency vs δ  
- `patch_shuffle_results.json` — accuracy drop + consistency  
- `feature_similarity.json` — cosine stability for required interventions  
- `task1_summary.json` — one-file dump of everything above  

Under `task1/results/figures/`:

- `translation_accuracy.png`, `translation_consistency.png`  
- `{backbone}_{grayscale|patch_shuffle}_tsne.png`  
- `representation_settings.json`  

## Attribution

- AdaIN architecture / weights: [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) (Huang & Belongie, 2017)  
- Backbones: `torchvision` ResNet-50 / ViT-B/16; OpenCLIP ViT-B-32 (`pretrained='openai'`)  
- Shape-bias metric definition: Geirhos et al. (2019)  
