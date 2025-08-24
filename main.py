import argparse
import os
import random

import numpy as np
import torch

from data_loader import build_dataloaders_from_synthetic, build_dataloaders_from_csv
from engine import TrainConfig, train_t_spad, eval_t_spad, train_hy_ltra, eval_hy_ltra


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Hy-LTra training and evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # common args
    def add_common_args(p):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--epochs", type=int, default=5)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--weight_decay", type=float, default=0.0)
        p.add_argument("--grad_clip", type=float, default=1.0)
        p.add_argument("--save_dir", type=str, default="checkpoints")
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--seed", type=int, default=42)

    # dataset selection
    def add_dataset_args(p):
        src = p.add_mutually_exclusive_group(required=False)
        src.add_argument("--synthetic", action="store_true", help="Use synthetic dataset")
        src.add_argument("--csv_path", type=str, default=None, help="Path to CSV with coordinates and AP columns")
        p.add_argument("--T", type=int, default=10)
        p.add_argument("--num_sequences_train", type=int, default=2000)
        p.add_argument("--num_sequences_val", type=int, default=200)
        p.add_argument("--sigma", type=float, default=5.0)
        p.add_argument("--noise_std", type=float, default=1.0)
        p.add_argument("--grid_w", type=int, default=8)
        p.add_argument("--grid_h", type=int, default=8)
        p.add_argument("--num_aps", type=int, default=32)
        p.add_argument("--ap_prefix", type=str, default=None, help="Optional AP column prefix for CSV")

    # train t_spad
    p_t = sub.add_parser("train_tspad", help="Train T-SPAD only")
    add_common_args(p_t)
    add_dataset_args(p_t)

    # train hy_ltra (L-TLM in framework with frozen T-SPAD)
    p_h = sub.add_parser("train_hyltra", help="Train Hy-LTra with frozen T-SPAD")
    add_common_args(p_h)
    add_dataset_args(p_h)
    p_h.add_argument("--t_spad_ckpt", type=str, default=os.path.join("checkpoints", "t_spad_best.pth"))

    # quick smoke test
    p_s = sub.add_parser("smoke", help="Run a quick synthetic smoke test for both phases")
    p_s.add_argument("--device", default="cpu")

    args = parser.parse_args()
    set_seed(args.seed if hasattr(args, "seed") else 42)

    if args.cmd == "smoke":
        print("[Smoke] Building tiny synthetic dataloaders…")
        train_loader, val_loader, num_aps = build_dataloaders_from_synthetic(
            T=6, num_sequences_train=64, num_sequences_val=32, grid_size=(5, 5), num_aps=16, sigma=2.5, noise_std=0.5, batch_size=16, seed=7
        )
        cfg = TrainConfig(device=args.device, epochs=1, lr=1e-3, grad_clip=1.0, save_dir="checkpoints")

        print("[Smoke] Phase 1: Train T-SPAD…")
        t_spad, best_t = train_t_spad(train_loader, val_loader, num_aps=num_aps, cfg=cfg)
        print("[Smoke] Best T-SPAD val:", best_t)

        print("[Smoke] Phase 2: Train Hy-LTra…")
        hy, best_h = train_hy_ltra(train_loader, val_loader, num_aps=num_aps, cfg=cfg, t_spad_ckpt=os.path.join("checkpoints", "t_spad_best.pth"))
        print("[Smoke] Best Hy-LTra val:", best_h)
        return

    # build dataloaders
    if getattr(args, "synthetic", False) or (not args.csv_path):
        train_loader, val_loader, num_aps = build_dataloaders_from_synthetic(
            T=args.T,
            num_sequences_train=args.num_sequences_train,
            num_sequences_val=args.num_sequences_val,
            grid_size=(args.grid_h, args.grid_w),
            num_aps=args.num_aps,
            sigma=args.sigma,
            noise_std=args.noise_std,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        ap_columns = None
        if args.ap_prefix:
            # 让 data_loader 自动基于前缀推断
            ap_columns = None
        train_loader, val_loader, num_aps = build_dataloaders_from_csv(
            csv_path=args.csv_path,
            ap_columns=ap_columns,
            T=args.T,
            num_sequences_train=args.num_sequences_train,
            num_sequences_val=args.num_sequences_val,
            sigma=args.sigma,
            noise_std=args.noise_std,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    cfg = TrainConfig(device=args.device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip, save_dir=args.save_dir)

    if args.cmd == "train_tspad":
        model, best = train_t_spad(train_loader, val_loader, num_aps=num_aps, cfg=cfg)
        print("Best val:", best)
    elif args.cmd == "train_hyltra":
        model, best = train_hy_ltra(train_loader, val_loader, num_aps=num_aps, cfg=cfg, t_spad_ckpt=args.t_spad_ckpt)
        print("Best val:", best)
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()

