"""Task 4 training: Vanilla, GCSC, PROSER.

Checkpoint rule: highest CIFAR-10 validation accuracy only.
No CIFAR-100 images are loaded in this file.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.seed import set_seed
from task4.data.cifar10 import build_cifar10_loaders
from task4.methods.proser import proser_batch_loss
from task4.models.resnet_cifar import build_resnet18_cifar


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_merged_config(base_path: Path, method_path: Path | None) -> dict:
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if method_path is not None:
        with open(method_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, override)
        method_name = cfg.get("method")
        if method_name and method_name in cfg.get("methods", {}):
            if method_name in override:
                cfg["methods"][method_name] = deep_update(
                    cfg["methods"][method_name], override[method_name]
                )
    return cfg


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        # PROSER CSA uses known-class logits only.
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.numel()
    return correct / max(total, 1)


def save_checkpoint(path: Path, model, cfg: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "meta": meta,
        "num_dummy": getattr(model, "num_dummy", 0),
    }
    torch.save(payload, path)


def train_closed_set(cfg: dict, device: torch.device) -> dict:
    """Train Vanilla or GCSC from random init."""
    method = cfg["method"]
    mcfg = cfg["methods"][method]
    use_ra = bool(mcfg.get("randaugment", method == "gcsc"))

    loaders = build_cifar10_loaders(cfg, randaugment=use_ra)
    model = build_resnet18_cifar(num_classes=cfg["model"]["num_classes"]).to(device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["max_epochs"]
    sched = CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_acc = -1.0
    best_path = (
        Path(cfg["paths"]["checkpoints_dir"]) / f"{method}_best.pt"
    )
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, labels, _ in tqdm(loaders["train"], desc=f"{method} ep{epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            running += float(loss.item()) * labels.size(0)
            n += labels.size(0)
        sched.step()

        train_loss = running / max(n, 1)
        val_acc = evaluate_accuracy(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(f"[{method}] epoch {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                best_path,
                model,
                cfg,
                meta={
                    "method": method,
                    "epoch": epoch,
                    "val_acc": best_acc,
                    "randaugment": use_ra,
                },
            )

    # Final test CSA for the selected checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_acc = evaluate_accuracy(model, loaders["test"], device)

    curves_dir = Path(cfg["paths"]["curves_dir"])
    curves_dir.mkdir(parents=True, exist_ok=True)
    hist_path = curves_dir / f"{method}_history.json"
    summary = {
        "method": method,
        "best_val_acc": best_acc,
        "test_csa": test_acc,
        "epochs": epochs,
        "seconds": time.time() - t0,
        "checkpoint": str(best_path).replace("\\", "/"),
        "history": history,
    }
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / f"{method}_train_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "history"}, f, indent=2)

    print(f"[{method}] best val_acc={best_acc:.4f} test_csa={test_acc:.4f} → {best_path}")
    return summary


def train_proser(cfg: dict, device: torch.device) -> dict:
    """Fine-tune PROSER from the selected Vanilla checkpoint."""
    method = "proser"
    mcfg = cfg["methods"]["proser"]
    vanilla_ckpt = Path(cfg["paths"]["checkpoints_dir"]) / "vanilla_best.pt"
    if not vanilla_ckpt.exists():
        raise SystemExit(f"Missing Vanilla checkpoint: {vanilla_ckpt}")

    loaders = build_cifar10_loaders(cfg, randaugment=False)
    model = build_resnet18_cifar(num_classes=cfg["model"]["num_classes"])
    ckpt = torch.load(vanilla_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.attach_dummy_heads(num_dummy=int(mcfg.get("num_dummy", 5)))
    model = model.to(device)

    epochs = int(mcfg.get("max_epochs", 50))
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(mcfg.get("lr", 1e-3)),
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched = CosineAnnealingLR(opt, T_max=epochs)
    beta = float(mcfg.get("beta", 1.0))
    gamma = float(mcfg.get("gamma", 0.1))
    mixup_alpha = float(mcfg.get("mixup_alpha", 2.0))

    history = []
    best_acc = -1.0
    best_path = Path(cfg["paths"]["checkpoints_dir"]) / "proser_best.pt"
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, labels, _ in tqdm(loaders["train"], desc=f"proser ep{epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            out = proser_batch_loss(
                model,
                images,
                labels,
                beta=beta,
                gamma=gamma,
                mixup_alpha=mixup_alpha,
            )
            out["loss"].backward()
            opt.step()
            running += float(out["loss"].item()) * labels.size(0)
            n += labels.size(0)
        sched.step()

        train_loss = running / max(n, 1)
        val_acc = evaluate_accuracy(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(f"[proser] epoch {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                best_path,
                model,
                cfg,
                meta={
                    "method": method,
                    "epoch": epoch,
                    "val_acc": best_acc,
                    "init_from": str(vanilla_ckpt).replace("\\", "/"),
                    "beta": beta,
                    "gamma": gamma,
                    "num_dummy": model.num_dummy,
                },
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_acc = evaluate_accuracy(model, loaders["test"], device)

    curves_dir = Path(cfg["paths"]["curves_dir"])
    curves_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": method,
        "best_val_acc": best_acc,
        "test_csa": test_acc,
        "epochs": epochs,
        "seconds": time.time() - t0,
        "checkpoint": str(best_path).replace("\\", "/"),
        "history": history,
    }
    with open(curves_dir / "proser_history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tables_dir = Path(cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "proser_train_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "history"}, f, indent=2)

    print(f"[proser] best val_acc={best_acc:.4f} test_csa={test_acc:.4f} → {best_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method-config",
        type=str,
        required=True,
        help="e.g. task4/configs/vanilla.yaml",
    )
    parser.add_argument("--base-config", type=str, default="task4/configs/base.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_merged_config(Path(args.base_config), Path(args.method_config))
    set_seed(int(cfg["seed"]))
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device} method={cfg['method']}")

    if cfg["method"] in ("vanilla", "gcsc"):
        train_closed_set(cfg, device)
    elif cfg["method"] == "proser":
        train_proser(cfg, device)
    else:
        raise SystemExit(f"Unknown method: {cfg['method']}")


if __name__ == "__main__":
    main()
