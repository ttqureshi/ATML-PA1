"""Task 1 end-to-end runner.

Runs the assignment steps in order:
  1) Build splits / eval subset
  2) Train linear heads on frozen backbones
  3) Generate AdaIN cue-conflict images
  4) Evaluate clean + interventions
  5) Representation cosine stability
  6) t-SNE visualizations
  7) Write a compact summary JSON for the report

Usage (from repository root):
  python -m task1.scripts.run_task1
  python -m task1.scripts.run_task1 --stages splits,train
  python -m task1.scripts.run_task1 --config task1/configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

# Make repository root importable when run as a file or module.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_translation_curves(tables_dir: Path, figures_dir: Path) -> None:
    path = tables_dir / "translation_results.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, rows in data.items():
        xs = [r["displacement"] for r in rows]
        ys = [r["prediction_consistency"] for r in rows]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Displacement δ (pixels)")
    plt.ylabel("Prediction consistency")
    plt.title("Translation consistency vs displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = figures_dir / "translation_consistency.png"
    plt.savefig(out, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, rows in data.items():
        xs = [r["displacement"] for r in rows]
        ys = [r["top1_accuracy"] for r in rows]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Displacement δ (pixels)")
    plt.ylabel("Top-1 accuracy")
    plt.title("Translation accuracy vs displacement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = figures_dir / "translation_accuracy.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote translation plots to {figures_dir}")


def write_report_summary(cfg: dict) -> None:
    """Compact JSON the report writer can cite directly."""
    tables = Path(cfg["paths"]["tables_dir"])
    summary = {
        "assumptions": {
            "dataset": "STL-10",
            "seed": cfg["seed"],
            "extra_color_intervention": "hue_rotate",
            "hue_degrees": cfg["color"]["hue_degrees"],
            "cue_conflict_method": "AdaIN",
            "style_alpha": cfg["cue_conflict"]["style_alpha"],
            "class_pairs": cfg["cue_conflict"]["class_pairs"],
            "representation_viz": cfg["representation"]["method"],
        },
        "clean": _read_json(tables / "clean_baseline.json"),
        "color": _read_json(tables / "color_results.json"),
        "patch_shuffle": _read_json(tables / "patch_shuffle_results.json"),
        "cue_conflict": _read_json(tables / "cue_conflict_results.json"),
        "translation": _read_json(tables / "translation_results.json"),
        "feature_similarity": _read_json(tables / "feature_similarity.json"),
    }
    out = tables / "task1_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 pipeline.")
    parser.add_argument("--config", default="task1/configs/default.yaml")
    parser.add_argument(
        "--stages",
        default="all",
        help=(
            "Comma-separated stages: splits,train,cues,eval,features,repr,summary "
            "or 'all'."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    stages = {s.strip() for s in args.stages.split(",")}
    if "all" in stages:
        stages = {"splits", "train", "cues", "eval", "features", "repr", "summary"}

    if "splits" in stages:
        from task1.data.make_subset import main as make_subset_main

        print("\n===== STAGE: splits =====")
        make_subset_main(args.config)

    if "train" in stages:
        from task1.models.train_heads import train_all_heads

        print("\n===== STAGE: train linear heads =====")
        train_all_heads(args.config)

    if "cues" in stages:
        from task1.data.make_cue_conflicts import main as make_cues_main

        print("\n===== STAGE: cue conflicts =====")
        make_cues_main(args.config)

    if "eval" in stages:
        from task1.analysis.evaluate_bias import run_all_evaluations

        print("\n===== STAGE: evaluate interventions =====")
        run_all_evaluations(args.config)
        plot_translation_curves(
            Path(cfg["paths"]["tables_dir"]),
            Path(cfg["paths"]["figures_dir"]),
        )

    if "features" in stages:
        from task1.analysis.feature_similarity import run_feature_similarity

        print("\n===== STAGE: feature similarity =====")
        run_feature_similarity(args.config)

    if "repr" in stages:
        from task1.analysis.representation import run_representation_analysis

        print("\n===== STAGE: representation visualization =====")
        run_representation_analysis(args.config)

    if "summary" in stages:
        print("\n===== STAGE: summary =====")
        write_report_summary(cfg)

    print("\nTask 1 pipeline finished.")


if __name__ == "__main__":
    main()
