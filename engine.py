from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import T_SPAD, Hy_LTra


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    save_dir: str = "checkpoints"


# -----------------------------
# Metrics
# -----------------------------
def mse_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean()


def localization_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: (B, T, C)
    return ((pred - target) ** 2).mean()


# -----------------------------
# T-SPAD 训练/评估
# -----------------------------
def train_t_spad(train_loader: DataLoader, val_loader: DataLoader, num_aps: int, cfg: TrainConfig) -> Tuple[T_SPAD, Dict[str, float]]:
    device = torch.device(cfg.device)
    model = T_SPAD(num_aps=num_aps).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = {"val_loss": float("inf")}
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"T-SPAD Train {epoch}/{cfg.epochs}")
        total = 0.0
        n = 0
        for batch in pbar:
            rssi_seq, coord_seq, prev_coords = batch
            rssi_seq = rssi_seq.to(device)  # (B, T, M)
            B, T, M = rssi_seq.shape
            loss = 0.0
            for t in range(T - 1):
                pred = model(rssi_seq[:, : t + 1, :])      # (B, M)
                target = rssi_seq[:, t + 1, :]             # (B, M)
                loss = loss + mse_loss(pred, target)
            loss = loss / max(T - 1, 1)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * B
            n += B
            pbar.set_postfix(loss=total / n)

        val = eval_t_spad(val_loader, model, device)
        if val["val_loss"] < best["val_loss"]:
            best = val
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "t_spad_best.pth"))

    return model, best


def eval_t_spad(loader: DataLoader, model: T_SPAD, device: Optional[torch.device] = None) -> Dict[str, float]:
    device = torch.device("cpu") if device is None else torch.device(device)
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            rssi_seq, coord_seq, prev_coords = batch
            rssi_seq = rssi_seq.to(device)
            B, T, M = rssi_seq.shape
            loss = 0.0
            for t in range(T - 1):
                pred = model(rssi_seq[:, : t + 1, :])      # (B, M)
                target = rssi_seq[:, t + 1, :]
                loss = loss + mse_loss(pred, target)
            loss = loss / max(T - 1, 1)
            total += loss.item() * B
            n += B
    return {"val_loss": total / max(n, 1)}


# -----------------------------
# Hy-LTra(L-TLM) 训练/评估（冻结T-SPAD）
# -----------------------------
def train_hy_ltra(train_loader: DataLoader, val_loader: DataLoader, num_aps: int, cfg: TrainConfig, t_spad_ckpt: Optional[str] = None) -> Tuple[Hy_LTra, Dict[str, float]]:
    device = torch.device(cfg.device)
    model = Hy_LTra(num_aps=num_aps).to(device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 加载并冻结 T-SPAD
    if t_spad_ckpt and os.path.isfile(t_spad_ckpt):
        state = torch.load(t_spad_ckpt, map_location=device)
        model.t_spad.load_state_dict(state, strict=False)
    model.freeze_t_spad()

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = {"val_loss": float("inf")}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Hy-LTra Train {epoch}/{cfg.epochs}")
        total = 0.0
        n = 0
        for batch in pbar:
            rssi_seq, coord_seq, prev_coords = batch
            rssi_seq = rssi_seq.to(device)
            coord_seq = coord_seq.to(device)
            prev_coords = prev_coords.to(device)

            pred_traj, gating = model(rssi_seq, prev_coords)
            loss = localization_mse(pred_traj, coord_seq)

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), cfg.grad_clip)
            opt.step()

            total += loss.item() * rssi_seq.size(0)
            n += rssi_seq.size(0)
            pbar.set_postfix(loss=total / n)

        val = eval_hy_ltra(val_loader, model, device)
        if val["val_loss"] < best["val_loss"]:
            best = val
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "hy_ltra_best.pth"))

    return model, best


def eval_hy_ltra(loader: DataLoader, model: Hy_LTra, device: Optional[torch.device] = None) -> Dict[str, float]:
    device = torch.device("cpu") if device is None else torch.device(device)
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            rssi_seq, coord_seq, prev_coords = batch
            rssi_seq = rssi_seq.to(device)
            coord_seq = coord_seq.to(device)
            prev_coords = prev_coords.to(device)
            pred_traj, gating = model(rssi_seq, prev_coords)
            loss = localization_mse(pred_traj, coord_seq)
            total += loss.item() * rssi_seq.size(0)
            n += rssi_seq.size(0)
    return {"val_loss": total / max(n, 1)}
