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
| 2 Unsupervised DA | Pending | — |
| 3 Domain generalization | Pending | — |
| 4 Open-set recognition | Pending | — |

See [`task1/README.md`](task1/README.md) for Task 1 details, assumptions, and report outputs.

## Reproducibility

- Global seed: **6304**
- Do not commit datasets or large checkpoints (see `.gitignore`)
- Keep machine-readable JSON results under each task's `results/` folder
