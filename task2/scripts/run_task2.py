"""Task 2 stage runner (mirrors task1/scripts/run_task1.py style).

Stages:
  splits          — verify PACS layout + write shared split JSON
  train_source    — Source-only ERM (Task 3 baseline checkpoint)
  train_dan       — DAN with λ_MMD=1
  train_dann      — DANN
  train_cdan      — CDAN
  train_main      — all four main methods in order
  study_lambda    — controlled DAN λ study {0.1, 1, 10}
  eval            — final Sketch + domain-separability + class analysis

Do NOT run long GPU stages until you explicitly ask to execute on Colab.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        type=str,
        default="splits",
        help="Comma-separated stages (see module docstring).",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    py = sys.executable
    device_args = ["--device", args.device] if args.device else []

    for stage in stages:
        if stage == "splits":
            run([py, "-m", "shared.prepare_pacs_splits"])
        elif stage == "train_source":
            run(
                [
                    py,
                    "-m",
                    "task2.train",
                    "--method-config",
                    "task2/configs/source_only.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_dan":
            run(
                [
                    py,
                    "-m",
                    "task2.train",
                    "--method-config",
                    "task2/configs/dan.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_dann":
            run(
                [
                    py,
                    "-m",
                    "task2.train",
                    "--method-config",
                    "task2/configs/dann.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_cdan":
            run(
                [
                    py,
                    "-m",
                    "task2.train",
                    "--method-config",
                    "task2/configs/cdan.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_main":
            for s in ("train_source", "train_dan", "train_dann", "train_cdan"):
                # Recurse via subprocess for clear logs.
                run([py, "-m", "task2.scripts.run_task2", "--stages", s, *device_args])
        elif stage == "study_lambda":
            for lam in (0.1, 1.0, 10.0):
                run(
                    [
                        py,
                        "-m",
                        "task2.train",
                        "--method-config",
                        "task2/configs/dan.yaml",
                        "--lambda-mmd",
                        str(lam),
                        *device_args,
                    ]
                )
        elif stage == "eval":
            run([py, "-m", "task2.evaluate_final", *device_args])
        else:
            raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
