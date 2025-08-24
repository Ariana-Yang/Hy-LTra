import math
import random
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 滤波器：迭代递归加权平均(指数平滑)
# -----------------------------
def exponential_smoothing(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    对一维或二维信号进行指数加权平滑。
    - 若输入为 shape (T,), 返回相同形状。
    - 若输入为 shape (T, M), 将对每列独立平滑，返回 (T, M)。
    """
    x = np.asarray(signal)
    if x.ndim == 1:
        y = np.zeros_like(x, dtype=float)
        y[0] = x[0]
        for t in range(1, len(x)):
            y[t] = (1 - alpha) * y[t - 1] + alpha * x[t]
        return y
    elif x.ndim == 2:
        T, M = x.shape
        y = np.zeros_like(x, dtype=float)
        y[0, :] = x[0, :]
        for t in range(1, T):
            y[t, :] = (1 - alpha) * y[t - 1, :] + alpha * x[t, :]
        return y
    else:
        raise ValueError("signal must be 1D or 2D array")


# -----------------------------
# 轨迹生成：基于高斯转移概率的随机游走
# -----------------------------
def pairwise_dist(coords: np.ndarray) -> np.ndarray:
    """ 欧氏距离矩阵，coords: (N, D) -> (N, N) """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def gaussian_transition_matrix(dist_mat: np.ndarray, sigma: float) -> np.ndarray:
    """
    根据距离矩阵生成基于高斯核的转移概率矩阵 P，按行归一化。
    P[i, j] ∝ exp( - d(i,j)^2 / (2*sigma^2) )，且 P[i,i] 设为一个较小但非零的值以允许停留。
    """
    with np.errstate(over='ignore'):
        P = np.exp(-(dist_mat ** 2) / (2 * (sigma ** 2) + 1e-8))
    # 允许停留：防止对角为1导致退化，稍微抑制自环
    np.fill_diagonal(P, np.diag(P) * 0.5 + 1e-6)
    # 行归一化
    row_sums = P.sum(axis=1, keepdims=True) + 1e-12
    P /= row_sums
    return P


def sample_trajectory(P: np.ndarray, T: int, start_idx: Optional[int] = None, rng: Optional[random.Random] = None) -> List[int]:
    """
    从转移矩阵 P 采样长度为 T 的状态序列(索引)。
    """
    rng = rng or random.Random()
    N = P.shape[0]
    if start_idx is None:
        cur = rng.randrange(N)
    else:
        cur = int(start_idx)
    traj = [cur]
    for _ in range(1, T):
        probs = P[cur]
        next_idx = rng.choices(range(N), weights=probs, k=1)[0]
        traj.append(next_idx)
        cur = next_idx
    return traj


class TrajectoryDataset(Dataset):
    """
    生成轨迹序列数据：每个样本包含
    - rssi_seq: (T, M)
    - coord_seq: (T, D)
    - prev_coords: (T, D) 供 L-TLM 使用（teacher forcing 可由上层控制）

    输入:
    - fingerprints: (N, M) 每个参考点的RSSI指纹
    - coords: (N, D) 每个参考点的坐标
    - T: 时间步长度
    - num_sequences: 样本数量
    - sigma: 高斯转移核带宽
    - noise_std: 生成序列时对RSSI添加的小噪声（可选）
    """
    def __init__(
        self,
        fingerprints: np.ndarray,
        coords: np.ndarray,
        T: int = 10,
        num_sequences: int = 1000,
        sigma: float = 5.0,
        noise_std: float = 1.0,
        use_filter: bool = True,
        filter_alpha: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert fingerprints.ndim == 2 and coords.ndim == 2
        assert fingerprints.shape[0] == coords.shape[0]
        self.fingerprints = fingerprints.astype(float)
        self.coords = coords.astype(float)
        self.T = int(T)
        self.num_sequences = int(num_sequences)
        self.noise_std = float(noise_std)
        self.rng = random.Random(seed)
        # 预计算转移概率
        dist = pairwise_dist(self.coords)
        self.P = gaussian_transition_matrix(dist, sigma=sigma)
        # 可选滤波
        if use_filter:
            self.fingerprints = np.stack([
                exponential_smoothing(self.fingerprints[:, i], alpha=filter_alpha)
                for i in range(self.fingerprints.shape[1])
            ], axis=1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        traj_idx = sample_trajectory(self.P, self.T, rng=self.rng)
        rssi_seq = self.fingerprints[traj_idx, :]  # (T, M)
        if self.noise_std > 0:
            rssi_seq = rssi_seq + np.random.normal(0, self.noise_std, size=rssi_seq.shape)
        coord_seq = self.coords[traj_idx, :]       # (T, D)
        prev_coords = np.vstack([
            np.zeros_like(coord_seq[0:1, :]),
            coord_seq[:-1, :]
        ])
        # to torch
        rssi_seq = torch.tensor(rssi_seq, dtype=torch.float32)
        coord_seq = torch.tensor(coord_seq, dtype=torch.float32)
        prev_coords = torch.tensor(prev_coords, dtype=torch.float32)
        return rssi_seq, coord_seq, prev_coords


# -----------------------------
# 合成数据生成：用于快速自检
# -----------------------------
def make_synthetic_fingerprints(
    grid_size: Tuple[int, int] = (10, 10),
    num_aps: int = 32,
    rssi_min: int = -95,
    rssi_max: int = -30,
    pathloss_sigma: float = 8.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成简单的合成RSSI指纹：
    - 在一个 grid 上放置 RPs，坐标为整数网格点
    - 随机放置 APs 并按距离产生RSSI（对数路径损耗的粗略近似），裁剪至[rssi_min, rssi_max]
    返回:
    fingerprints: (N, M)
    coords: (N, 2)
    """
    rng = np.random.default_rng(seed)
    H, W = grid_size
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    N = coords.shape[0]

    # 随机放置 AP
    ap_pos = rng.uniform(low=[0, 0], high=[W - 1, H - 1], size=(num_aps, 2))

    # 距离 -> RSSI 粗略模型: rssi = rssi_max - 10*n*log10(d+1) + noise
    n = 2.2
    eps = 1e-3
    fingerprints = np.zeros((N, num_aps), dtype=float)
    for i, rp in enumerate(coords):
        dists = np.sqrt(((ap_pos - rp) ** 2).sum(axis=1)) + eps
        rssi = rssi_max - 10 * n * np.log10(dists + 1.0)
        rssi += rng.normal(0, pathloss_sigma, size=rssi.shape)
        fingerprints[i, :] = np.clip(rssi, rssi_min, rssi_max)

    return fingerprints, coords


def build_dataloaders_from_synthetic(
    T: int = 10,
    num_sequences_train: int = 2000,
    num_sequences_val: int = 200,
    grid_size: Tuple[int, int] = (8, 8),
    num_aps: int = 32,
    sigma: float = 3.0,
    noise_std: float = 1.0,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int]:
    """构造合成数据的训练/验证 DataLoader，并返回 AP 数量。"""
    fp, coords = make_synthetic_fingerprints(grid_size=grid_size, num_aps=num_aps, seed=seed)
    train_ds = TrajectoryDataset(fp, coords, T=T, num_sequences=num_sequences_train, sigma=sigma, noise_std=noise_std, seed=seed)
    val_ds = TrajectoryDataset(fp, coords, T=T, num_sequences=num_sequences_val, sigma=sigma, noise_std=noise_std, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, fp.shape[1]


# -----------------------------
# 占位：从真实数据构建 DataLoader（可后续接入 UJIIndoorLoc / SODIndoorLoc）
# -----------------------------
def build_dataloaders_from_csv(
    csv_path: str,
    ap_columns: Optional[List[str]] = None,
    x_col: str = "x",
    y_col: str = "y",
    T: int = 10,
    num_sequences_train: int = 2000,
    num_sequences_val: int = 200,
    sigma: float = 5.0,
    noise_std: float = 1.0,
    batch_size: int = 64,
    seed: int = 42,
):
    """
    从一个包含坐标与RSSI的CSV构建 DataLoader。
    期望: 每行代表一个 RP，包含 x,y 与若干 AP 列(ap_columns)。
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if ap_columns is None:
        ap_columns = [c for c in df.columns if c.lower().startswith("ap") or c.upper().startswith("WAP")]
        if len(ap_columns) == 0:
            raise ValueError("无法自动识别 AP 列，请显式传入 ap_columns")

    coords = df[[x_col, y_col]].values.astype(float)
    fingerprints = df[ap_columns].values.astype(float)

    # 缺失值处理：未检测AP设为-100
    fingerprints = np.nan_to_num(fingerprints, nan=-100.0)

    train_ds = TrajectoryDataset(fingerprints, coords, T=T, num_sequences=num_sequences_train, sigma=sigma, noise_std=noise_std, seed=seed)
    val_ds = TrajectoryDataset(fingerprints, coords, T=T, num_sequences=num_sequences_val, sigma=sigma, noise_std=noise_std, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, fingerprints.shape[1]

