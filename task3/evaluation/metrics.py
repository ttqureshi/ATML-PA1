"""Small metric helpers for Task 3 tables (mean / worst-domain)."""

from __future__ import annotations


def with_worst_domain(source_metrics: dict) -> dict:
    """Add worst-domain Acc / macro-F1 to an ``evaluate_sources`` result dict."""
    per = source_metrics["per_domain"]
    worst_f1_domain = min(per, key=lambda d: per[d]["macro_f1"])
    worst_acc_domain = min(per, key=lambda d: per[d]["accuracy"])
    out = dict(source_metrics)
    out["worst_macro_f1"] = float(per[worst_f1_domain]["macro_f1"])
    out["worst_macro_f1_domain"] = worst_f1_domain
    out["worst_accuracy"] = float(per[worst_acc_domain]["accuracy"])
    out["worst_accuracy_domain"] = worst_acc_domain
    # Keep means as floats for JSON friendliness.
    out["mean_macro_f1"] = float(source_metrics["mean_macro_f1"])
    out["mean_accuracy"] = float(source_metrics["mean_accuracy"])
    return out


def summarize_per_domain(source_metrics: dict) -> dict:
    """Compact Acc/F1 per domain for tables."""
    return {
        d: {
            "accuracy": float(m["accuracy"]),
            "macro_f1": float(m["macro_f1"]),
        }
        for d, m in source_metrics["per_domain"].items()
    }
