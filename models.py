from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class T_SPAD(nn.Module):
    """
    Transformer-based Signal Prediction & Anomaly Detection
    输入: rssi_sequence (B, T, M)
    输出: next_rssi (B, M) 预测下一个时刻的RSSI
    """
    def __init__(self, num_aps: int, embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 2, max_len: int = 512):
        super().__init__()
        self.num_aps = num_aps
        # 卷积块：沿时间维提取局部特征
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_aps, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc_embed = nn.Linear(128, embedding_dim)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, num_aps)

    @staticmethod
    def causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # 允许自注意与过去，不允许看未来
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, rssi_sequence: torch.Tensor) -> torch.Tensor:
        # rssi_sequence: (B, T, M)
        B, T, M = rssi_sequence.shape
        assert M == self.num_aps, f"Expected num_aps={self.num_aps}, got {M}"
        x = rssi_sequence.permute(0, 2, 1)  # (B, M, T)
        x = self.conv_block(x)              # (B, 128, T)
        x = x.permute(0, 2, 1)              # (B, T, 128)
        x = self.fc_embed(x)                # (B, T, D)
        x = self.pos_enc(x)
        mask = self.causal_mask(T, x.device)
        h = self.transformer_encoder(x, mask=mask)  # (B, T, D)
        last = h[:, -1, :]                 # (B, D)
        next_rssi = self.output_layer(last) # (B, M)
        return next_rssi


class L_TLM(nn.Module):
    """
    LSTM-based Trajectory Localization Module (P-MIMO)
    输入:
    - rssi_features: (B, T, M)
    - prev_coords: (B, T, C)
    输出:
    - predicted_coords: (B, T, C)
    """
    def __init__(self, num_aps: int, coord_dims: int = 2, hidden_size: int = 100, num_layers: int = 2):
        super().__init__()
        self.input_size = num_aps + coord_dims
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_output = nn.Linear(hidden_size, coord_dims)

    def forward(self, rssi_features: torch.Tensor, prev_coords: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rssi_features, prev_coords], dim=2)  # (B, T, M+C)
        h, _ = self.lstm(x)
        out = self.fc_output(h)
        return out


class Hy_LTra(nn.Module):
    """
    Hy-LTra 主框架，内含 T-SPAD 与 L-TLM，并实现 A-AFM 融合。
    forward 输入:
    - rssi_sequence: (B, T, M)
    - prev_coords_sequence: (B, T, C)
    返回:
    - predicted_trajectory: (B, T, C)
    - gating_coeffs: (B, T, 1)
    """
    def __init__(self, num_aps: int, coord_dims: int = 2, beta: float = 0.5, t_spad_kwargs: Optional[dict] = None, l_tlm_kwargs: Optional[dict] = None):
        super().__init__()
        t_spad_kwargs = t_spad_kwargs or {}
        l_tlm_kwargs = l_tlm_kwargs or {}
        self.t_spad = T_SPAD(num_aps=num_aps, **t_spad_kwargs)
        self.l_tlm = L_TLM(num_aps=num_aps, coord_dims=coord_dims, **l_tlm_kwargs)
        self.beta = beta
        self.coord_dims = coord_dims
        self.num_aps = num_aps

    @torch.no_grad()
    def compute_anomaly_scores(self, rssi_sequence: torch.Tensor) -> torch.Tensor:
        """
        基于 T-SPAD 的一步预测误差得到异常分数 A_t，返回 (B, T) 并在 t=0 处补0。
        """
        B, T, M = rssi_sequence.shape
        device = rssi_sequence.device
        scores = []
        for t in range(T - 1):
            prefix = rssi_sequence[:, : t + 1, :]
            pred_next = self.t_spad(prefix)          # (B, M)
            actual = rssi_sequence[:, t + 1, :]      # (B, M)
            mse = F.mse_loss(pred_next, actual, reduction='none').mean(dim=1)  # (B,)
            scores.append(mse)
        if len(scores) == 0:
            return torch.zeros(B, T, device=device)
        S = torch.stack(scores, dim=1)  # (B, T-1)
        S = F.pad(S, (1, 0), value=0.0)  # (B, T)
        return S

    def forward(self, rssi_sequence: torch.Tensor, prev_coords_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, M = rssi_sequence.shape
        # 1) T-SPAD 异常评分
        anomaly_scores = self.compute_anomaly_scores(rssi_sequence)  # (B, T)
        # 2) A-AFM 门控
        gating = torch.exp(-self.beta * anomaly_scores).unsqueeze(-1)  # (B, T, 1)
        # 3) 门控后的RSSI
        gated_rssi = rssi_sequence * gating
        # 4) L-TLM 轨迹预测
        traj = self.l_tlm(gated_rssi, prev_coords_sequence)
        return traj, gating

    def freeze_t_spad(self):
        for p in self.t_spad.parameters():
            p.requires_grad = False
        self.t_spad.eval()

    def unfreeze_t_spad(self):
        for p in self.t_spad.parameters():
            p.requires_grad = True
        self.t_spad.train()
