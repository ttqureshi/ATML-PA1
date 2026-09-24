"""Helpers to download / verify the PACS dataset layout.

PACS is not bundled with torchvision. We expect the standard folder layout::

    data/pacs/{photo,art_painting,cartoon,sketch}/{class}/*.jpg

The Colab notebook documents a practical download path (gdown / zip).
This module only *verifies* the layout and builds the shared split file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.pacs import (  # noqa: E402
    CLASS_NAMES,
    DOMAINS,
    build_source_splits,
    discover_pacs,
    save_splits,
)


DEFAULT_ROOT = ROOT / "data" / "pacs"
DEFAULT_SPLIT = ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"


def verify_layout(root: Path) -> dict:
    """Return counts per domain/class; raise if anything required is missing."""
    examples = discover_pacs(root)
    counts: dict[str, dict[str, int]] = {d: {c: 0 for c in CLASS_NAMES} for d in DOMAINS}
    for ex in examples:
        counts[ex.domain][ex.class_name] += 1
    return {
        "n_total": len(examples),
        "per_domain_class": counts,
        "per_domain": {d: sum(counts[d].values()) for d in DOMAINS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PACS and write shared splits.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-out", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    summary = verify_layout(args.root)
    print("PACS layout OK")
    print(f"  root: {args.root}")
    print(f"  n_total: {summary['n_total']}")
    for domain, n in summary["per_domain"].items():
        print(f"  {domain}: {n}")

    examples = discover_pacs(args.root)
    splits = build_source_splits(examples, train_frac=args.train_frac, seed=args.seed)
    save_splits(splits, args.split_out)
    print(f"Wrote splits -> {args.split_out}")


if __name__ == "__main__":
    main()
