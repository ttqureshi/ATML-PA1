"""Task 3 stage runner.

Stages:
  splits          — verify PACS layout + write shared split JSON (same as Task 2)
  check_erm       — verify Task 2 Source-only checkpoint exists (load-only ERM)
  train_dan_dg    — DAN-DG with λ_DG=1
  train_sam       — SAM with ρ=0.05
  train_main      — DAN-DG + SAM (ERM is load-only)
  study_lambda    — controlled DAN-DG λ study {0.1, 1, 10}
  eval_source     — source Acc/F1 + separability + sharpness (NO Sketch)
  eval            — final Sketch + class analysis (after decisions fixed)

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
        default="check_erm",
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
        elif stage == "check_erm":
            ckpt = ROOT / "task2" / "results" / "checkpoints" / "source_only_best.pt"
            if not ckpt.exists():
                raise SystemExit(
                    f"Missing ERM checkpoint: {ckpt}\n"
                    "Restore from Drive / prior Colab. Do NOT retrain under a new config."
                )
            print(f"OK: ERM checkpoint present ({ckpt.stat().st_size} bytes)")
        elif stage == "train_dan_dg":
            run(
                [
                    py,
                    "-m",
                    "task3.train",
                    "--method-config",
                    "task3/configs/dan_dg.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_sam":
            run(
                [
                    py,
                    "-m",
                    "task3.train",
                    "--method-config",
                    "task3/configs/sam.yaml",
                    *device_args,
                ]
            )
        elif stage == "train_main":
            for s in ("check_erm", "train_dan_dg", "train_sam"):
                run([py, "-m", "task3.scripts.run_task3", "--stages", s, *device_args])
        elif stage == "study_lambda":
            for lam in (0.1, 1.0, 10.0):
                run(
                    [
                        py,
                        "-m",
                        "task3.train",
                        "--method-config",
                        "task3/configs/dan_dg.yaml",
                        "--lambda-dg",
                        str(lam),
                        *device_args,
                    ]
                )
        elif stage == "eval_source":
            run([py, "-m", "task3.evaluate_source", *device_args])
        elif stage == "eval":
            run([py, "-m", "task3.evaluate_final", *device_args])
        else:
            raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
