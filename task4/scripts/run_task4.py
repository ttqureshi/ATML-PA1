"""Task 4 stage runner.

Stages:
  splits           — stratified CIFAR-10 90/10 JSON (seed 6304)
  train_vanilla    — Vanilla CE ResNet-18
  train_gcsc       — GCSC = Vanilla + RandAugment(2, 9)
  train_proser     — PROSER fine-tune from Vanilla
  train_main       — vanilla + gcsc + proser
  extract_vanilla / extract_gcsc / extract_proser
  extract_all      — extract all three
  eval             — scores + model comparison + figures + failures
  scores           — Vanilla post-hoc score table only
  models           — trained-model comparison only
  figures          — score-distribution figure
  failures         — Vanilla MLS false-accept examples

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
            run(
                [
                    py,
                    "-m",
                    "task4.data.make_splits",
                    "--root",
                    "data/cifar",
                    "--out",
                    "task4/results/splits/cifar10_seed6304.json",
                ]
            )
        elif stage == "train_vanilla":
            run(
                [
                    py,
                    "-m",
                    "task4.train",
                    "--method-config",
                    "task4/configs/vanilla.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_gcsc":
            run(
                [
                    py,
                    "-m",
                    "task4.train",
                    "--method-config",
                    "task4/configs/gcsc.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_proser":
            run(
                [
                    py,
                    "-m",
                    "task4.train",
                    "--method-config",
                    "task4/configs/proser.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_main":
            for s in ("train_vanilla", "train_gcsc", "train_proser"):
                run([py, "-m", "task4.scripts.run_task4", "--stages", s, *device_args])
        elif stage.startswith("extract_"):
            method = stage.replace("extract_", "")
            if method == "all":
                for m in ("vanilla", "gcsc", "proser"):
                    run(
                        [
                            py,
                            "-m",
                            "task4.extract_outputs",
                            "--method",
                            m,
                            *device_args,
                        ]
                    )
            else:
                run(
                    [
                        py,
                        "-m",
                        "task4.extract_outputs",
                        "--method",
                        method,
                        *device_args,
                    ]
                )
        elif stage == "eval":
            run([py, "-m", "task4.evaluate_osr", "--stages", "scores,models,figures,failures"])
        elif stage in ("scores", "models", "figures", "failures"):
            run([py, "-m", "task4.evaluate_osr", "--stages", stage])
        else:
            raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
