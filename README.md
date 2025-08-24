# Hy-LTra: Hybrid LSTM + Transformer for RSSI Indoor Localization

本仓库实现 Hy-LTra 框架的最小可运行版本，包含：
- data_loader.py：数据加载、指数平滑滤波、基于高斯转移的随机轨迹生成、PyTorch Dataset/DataLoader 封装；
- models.py：T_SPAD（Transformer 预测下一时刻 RSSI）、L_TLM（P-MIMO LSTM 轨迹定位）、Hy_LTra（A-AFM 融合）；
- engine.py：两阶段训练与评估循环；
- main.py：命令行入口（两阶段训练 + 合成数据 smoke test）。

参考融合：
- Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization-main：使用其思想实现滤波与随机轨迹生成（参见 RNN+LSTM.pdf）。
- Transformer-Based-Outlier-Detector-main：基于其 signal_prediction_module.pdf 设计 Transformer 自回归预测结构（T-SPAD）。


## 安装

```bash
pip install -r requirements.txt
```


## 快速自检（合成数据）
运行一个极小数据集的两阶段训练，各 1 epoch，验证端到端流程是否可用：

```bash
python main.py smoke --device cpu
```

预期：输出 T-SPAD 与 Hy-LTra 各 1 个 epoch 的训练进度条与验证损失。


## 两阶段训练（默认使用合成数据）
- 阶段一：训练 T-SPAD（仅正常数据，本实现使用合成数据近似）
```bash
python main.py train_tspad --synthetic --epochs 5 --batch_size 64 --device cuda
```
- 阶段二：冻结 T-SPAD，在 Hy-LTra 中训练 L-TLM
```bash
python main.py train_hyltra --synthetic --epochs 5 --batch_size 64 --device cuda \
  --t_spad_ckpt checkpoints/t_spad_best.pth
```

可调参数（节选）：
- `--T` 轨迹长度，`--num_sequences_train/val` 样本数，`--sigma` 转移核带宽，`--noise_std` RSSI 噪声，`--num_aps` AP 数量；
- `--grid_h/w` 合成 RP 网格尺寸；
- 优化相关：`--epochs`、`--lr`、`--grad_clip`、`--weight_decay`、`--batch_size`。


## 使用真实数据（CSV 占位）
提供了 `build_dataloaders_from_csv` 占位接口（data_loader.py）：
- 期望 CSV 至少包含 `x, y` 坐标列和若干 AP 列（列名以 `ap` 或 `WAP` 开头可自动识别）；
- 未检测到 AP 的 RSSI 用 -100 填充；
- 轨迹通过基于 RP 间欧氏距离的高斯转移随机生成；
- 例：
```bash
python main.py train_tspad --csv_path path/to/your.csv --epochs 10
python main.py train_hyltra --csv_path path/to/your.csv --epochs 10 --t_spad_ckpt checkpoints/t_spad_best.pth
```

如需对接 UJIIndoorLoc / SODIndoorLoc，请先将其整理为上述 CSV 结构或在 data_loader 中新增专用解析逻辑。


## 设计与实现要点
- 滤波：`exponential_smoothing` 指数平滑，简洁稳定；
- 轨迹：`TrajectoryDataset` 内部基于高斯核的转移矩阵随机采样状态序列（参见 RNN+LSTM.pdf III-B2）；
- T-SPAD：卷积提取局部时序特征 + 位置编码 + TransformerEncoder，因果掩码保证自回归，输出下一时刻 RSSI；
- L-TLM（P-MIMO）：输入为 `[RSSI_t, prev_coord_t]`，输出 `coord_t`；训练时通过数据集提供的 `prev_coords` 实现 teacher forcing；
- A-AFM：`alpha_t = exp(-beta * A_t)`，其中 `A_t` 为一步预测 MSE；t=0 处补 0 分使 `alpha_0=1`；
- 两阶段：先训练 T-SPAD，再用其冻结权重训练 Hy-LTra（梯度仅流向 L-TLM）。


## 结果与检查
- 本地 smoke 测试（CPU）显示：
  - T-SPAD 1 epoch 验证 MSE 约在 1e3 量级（合成数据，尺度较大属正常）；
  - Hy-LTra 1 epoch 轨迹 MSE ~ 5-6；
 仅作流程连通性检查，非最终性能。


## 目录结构
```
.
├── data_loader.py
├── engine.py
├── main.py
├── models.py
├── requirements.txt
├── checkpoints/
│   ├── t_spad_best.pth
│   └── hy_ltra_best.pth
├── Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization-main/
└── Transformer-Based-Outlier-Detector-main/
```


## 后续改进建议
- 将 T-SPAD 的输入由“变长前缀”替换为“固定滑窗”并进行批量化展开，获得更高训练吞吐；
- 将 `prev_coords` 在推理时替换为模型滚动预测（训练时仍可使用 teacher forcing）；
- 集成真实数据解析与训练脚本参数化；
- 在输出序列上增加滑动窗口平滑（RNN+LSTM.pdf 所述）。

