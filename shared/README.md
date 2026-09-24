# Shared PACS protocol (Tasks 2 and 3)

This folder owns the **dataset, splits, transforms, and BatchNorm / batching
rules** that Tasks 2 and 3 must share. Method-specific losses live under
`task2/` (and later `task3/`), not here.

## What lives here

| File | Role |
|------|------|
| `pacs.py` | Domain/class constants, discovery, stratified splits, transforms, dataset |
| `pacs_protocol.py` | Freeze BN running stats; cyclic / domain-balanced loaders |
| `prepare_pacs_splits.py` | Verify layout + write `splits/pacs_sketch_seed6304.json` |
| `splits/pacs_sketch_seed6304.json` | Generated once; reused by Task 3 |

## PACS layout

```
data/pacs/
  photo/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
  art_painting/...
  cartoon/...
  sketch/...
```

`data/` is gitignored. On Colab, download then place domains at `data/pacs/`.

Public DomainBed-style archive (Google Drive id `1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd`):

```bash
pip install gdown
gdown 1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd -O data/PACS.zip
# unzip; move kfold/ contents so you have data/pacs/{photo,art_painting,cartoon,sketch}/
```

The Task 2 Colab notebook automates this. After the folders exist:

```bash
python -m shared.prepare_pacs_splits
```

## Hard rules (do not break)

1. **Sources:** Photo, Art Painting, Cartoon — labels used for CE + source-val selection.
2. **Target:** Sketch — Task 2 may use **images without class labels** during UDA.
3. **Task 3:** no Sketch until final eval; do not revise Task 3 using Task 2 Sketch scores.
4. **Splits:** stratified 80/20 per source domain, seed **6304**, written once.
5. **BN:** after `model.train()`, BN modules stay in `eval()` so running mean/var
   stay at ImageNet values; γ/β remain trainable.
6. **Batches (Task 2):** 8 images per source domain + 24 target images.
