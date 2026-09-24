"""Open-set evaluation from cached logits/features.

Produces:
  tables/task4_score_comparison.json   — Vanilla × {MSP, MLS, Energy, Mah}
  tables/task4_model_comparison.json   — Vanilla/GCSC/PROSER (+ placeholder)
  figures/score_distributions.png      — MSP / MLS / Mahalanobis panels
  tables/task4_failure_analysis.json   — Vanilla MLS false accepts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task4.evaluation.failure_analysis import collect_false_accepts, save_failures
from task4.evaluation.metrics import (
    auroc_known_vs_unknown,
    closed_set_accuracy,
    rejection_at_threshold,
)
from task4.evaluation.thresholds import threshold_from_val
from task4.scores import (
    energy_unknownness,
    fit_mahalanobis,
    mahalanobis_unknownness,
    mls_unknownness,
    msp_unknownness,
    proser_placeholder_unknownness,
)


def load_cache(cache_dir: Path, split: str) -> dict:
    path = cache_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_score(
    name: str,
    logits: np.ndarray,
    features: np.ndarray | None = None,
    mah_stats: tuple | None = None,
    dummy_logits: np.ndarray | None = None,
) -> np.ndarray:
    if name == "msp":
        return msp_unknownness(logits)
    if name == "mls":
        return mls_unknownness(logits)
    if name == "energy":
        return energy_unknownness(logits)
    if name == "mahalanobis":
        assert features is not None and mah_stats is not None
        means, inv_diag = mah_stats
        return mahalanobis_unknownness(features, means, inv_diag)
    if name == "proser_placeholder":
        assert dummy_logits is not None
        return proser_placeholder_unknownness(logits, dummy_logits)
    raise ValueError(name)


def eval_one_score(
    name: str,
    cache: dict,
    percentile: float,
    mah_stats: tuple | None = None,
) -> dict:
    val = cache["val"]
    test = cache["test"]
    near = cache["near"]
    far = cache["far"]

    kwargs = {}
    if name == "mahalanobis":
        kwargs["mah_stats"] = mah_stats
    if name == "proser_placeholder":
        pass

    def _u(split_name: str):
        sp = cache[split_name]
        return compute_score(
            name,
            sp["logits"],
            features=sp.get("features"),
            mah_stats=mah_stats,
            dummy_logits=sp.get("dummy_logits"),
        )

    u_val = _u("val")
    u_test = _u("test")
    u_near = _u("near")
    u_far = _u("far")
    u_all = np.concatenate([u_near, u_far])

    tau = threshold_from_val(u_val, percentile=percentile)
    rej = rejection_at_threshold(u_test, u_near, u_far, tau)

    csa = closed_set_accuracy(test["logits"], test["labels"])
    return {
        "score": name,
        "csa": csa,
        "auroc_near": auroc_known_vs_unknown(u_test, u_near),
        "auroc_far": auroc_known_vs_unknown(u_test, u_far),
        "auroc_all": auroc_known_vs_unknown(u_test, u_all),
        **rej,
        "u_test": u_test,
        "u_near": u_near,
        "u_far": u_far,
        "u_val": u_val,
        "pred_test": test["logits"].argmax(axis=1),
        "pred_near": near["logits"].argmax(axis=1),
        "pred_far": far["logits"].argmax(axis=1),
    }


def load_model_cache(cache_root: Path) -> dict:
    return {
        "train_eval": load_cache(cache_root, "train_eval"),
        "val": load_cache(cache_root, "val"),
        "test": load_cache(cache_root, "test"),
        "near": load_cache(cache_root, "near"),
        "far": load_cache(cache_root, "far"),
    }


def plot_score_distributions(
    vanilla_results: dict,
    out_path: Path,
) -> None:
    """Compact 3-panel score distributions: MSP, MLS, Mahalanobis."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, score in zip(axes, ["msp", "mls", "mahalanobis"]):
        r = vanilla_results[score]
        ax.hist(r["u_test"], bins=40, density=True, alpha=0.55, label="known test")
        ax.hist(r["u_near"], bins=40, density=True, alpha=0.55, label="near unk")
        ax.hist(r["u_far"], bins=40, density=True, alpha=0.55, label="far unk")
        ax.axvline(r["tau"], color="k", linestyle="--", linewidth=1, label=f"τ={r['tau']:.3g}")
        ax.set_title(score.upper())
        ax.set_xlabel("unknownness u(x)")
        ax.set_ylabel("density")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Vanilla post-hoc scores (larger u ⇒ more novel)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def strip_arrays(row: dict) -> dict:
    skip = {
        "u_test",
        "u_near",
        "u_far",
        "u_val",
        "pred_test",
        "pred_near",
        "pred_far",
    }
    return {k: v for k, v in row.items() if k not in skip}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="task4/configs/base.yaml")
    parser.add_argument(
        "--stages",
        type=str,
        default="scores,models,figures,failures",
        help="Comma-separated: scores, models, figures, failures",
    )
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["paths"]["cache_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    percentile = float(cfg["evaluation"]["threshold_percentile"])

    vanilla_cache = load_model_cache(cache_dir / "vanilla")
    mah_stats = fit_mahalanobis(
        vanilla_cache["train_eval"]["features"].astype(np.float64),
        vanilla_cache["train_eval"]["labels"],
        num_classes=10,
        eps=1e-6,
    )

    vanilla_score_results = {}

    if "scores" in stages:
        rows = []
        for score in ("msp", "mls", "energy", "mahalanobis"):
            print(f"Evaluating vanilla / {score}")
            r = eval_one_score(score, vanilla_cache, percentile, mah_stats=mah_stats)
            vanilla_score_results[score] = r
            rows.append({"model": "vanilla", **strip_arrays(r)})
        payload = {
            "description": "Post-hoc scores on frozen Vanilla model",
            "threshold_percentile": percentile,
            "rows": rows,
        }
        out = tables_dir / "task4_score_comparison.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out}")
    else:
        # Still need MLS for failures / figures if requested later
        for score in ("msp", "mls", "mahalanobis"):
            vanilla_score_results[score] = eval_one_score(
                score, vanilla_cache, percentile, mah_stats=mah_stats
            )

    if "models" in stages:
        rows = []
        # Vanilla + GCSC with MLS; PROSER with MLS and placeholder score.
        for method, score in (
            ("vanilla", "mls"),
            ("gcsc", "mls"),
            ("proser", "mls"),
            ("proser", "proser_placeholder"),
        ):
            print(f"Evaluating {method} / {score}")
            cache = load_model_cache(cache_dir / method)
            # Mahalanobis stats only needed for mah; not used here.
            r = eval_one_score(score, cache, percentile, mah_stats=None)
            rows.append({"model": method, **strip_arrays(r)})
            if method == "vanilla" and score == "mls":
                vanilla_score_results["mls"] = r

        # Attach CSA from each model's test logits (already in rows).
        payload = {
            "description": "Trained-model comparison (MLS common; PROSER placeholder extra)",
            "threshold_percentile": percentile,
            "rows": rows,
        }
        out = tables_dir / "task4_model_comparison.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out}")

    if "figures" in stages:
        if "msp" not in vanilla_score_results:
            for score in ("msp", "mls", "mahalanobis"):
                vanilla_score_results[score] = eval_one_score(
                    score, vanilla_cache, percentile, mah_stats=mah_stats
                )
        plot_score_distributions(
            vanilla_score_results,
            figures_dir / "score_distributions.png",
        )

    if "failures" in stages:
        if "mls" not in vanilla_score_results:
            vanilla_score_results["mls"] = eval_one_score(
                "mls", vanilla_cache, percentile, mah_stats=mah_stats
            )
        r = vanilla_score_results["mls"]
        near = vanilla_cache["near"]
        far = vanilla_cache["far"]
        near_rows = collect_false_accepts(
            u=r["u_near"],
            tau=r["tau"],
            pred_known=r["pred_near"],
            unknown_class_names=near["class_names"],
            group="near",
            top_k=3,
        )
        far_rows = collect_false_accepts(
            u=r["u_far"],
            tau=r["tau"],
            pred_known=r["pred_far"],
            unknown_class_names=far["class_names"],
            group="far",
            top_k=3,
        )
        all_rows = near_rows + far_rows
        out = tables_dir / "task4_failure_analysis.json"
        save_failures(all_rows, out)
        print(f"Wrote {out} ({len(all_rows)} examples)")


if __name__ == "__main__":
    main()
