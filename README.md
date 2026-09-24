# ATML PA1 — Beyond IID Learning

Programming Assignment 1 for EE-5102 / CS-6304 (Fall 2026).

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Tasks

| Task | Status | Entry point |
|------|--------|-------------|
| 1 Inductive biases | Implemented | `python -m task1.scripts.run_task1` |
| 2 Unsupervised DA | In progress | `python -m task2.scripts.run_task2` |
| 3 Domain generalization | Pending | (reuses `shared/` + Task 2 Source-only ckpt) |
| 4 Open-set recognition | Pending | — |

- Task 1: [`task1/README.md`](task1/README.md)
- Task 2: [`task2/README.md`](task2/README.md)
- Shared PACS protocol: [`shared/README.md`](shared/README.md)

## Reproducibility

- Global seed: **6304**
- Do not commit datasets or large checkpoints (see `.gitignore`)
- Keep machine-readable JSON results under each task's `results/` folder
