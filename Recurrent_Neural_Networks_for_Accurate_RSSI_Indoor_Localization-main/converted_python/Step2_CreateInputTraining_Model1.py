import numpy as np
import os

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Wifi Indoor Localization
# % Minhtu (Translated to Python)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# --- Configuration ---
# !!! 重要: 请将此路径修改为您自己的数据库文件夹路径 !!!
my_folder = r'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data'

# --- Load Data ---
print("Loading data...")
# 加载包含 X, Y 和 RSSI 读数的数据库
db_file = os.path.join(my_folder, 'UpdatedDatabase_8June2018_11MAC_AverageFilter.csv')
Database = np.loadtxt(db_file, delimiter=',')
L_Data = Database.shape[0]
NumReading = 11  # Number of RSSI columns (AP readings)

# 加载之前生成的轨迹文件
traj_file = os.path.join(my_folder, 'Traj_10points_5k_5m_s.csv')
TrajData = np.loadtxt(traj_file, delimiter=',')

# --- Get Dimensions ---
SizeTraj = TrajData.shape
L = SizeTraj[0]  # Number of trajectories
# 使用整数除法 //
NumPointPerTraj = SizeTraj[1] // 2

print(f"Number of trajectories: {L}")
print(f"Points per trajectory: {NumPointPerTraj}")

# -----------------------------------------------------------
# %%%% Add RSSI to the trajectory
# -----------------------------------------------------------

# 初始化用于存储 RNN 输入的数据库
# 注意：原始 MATLAB 代码初始化为 (NumPointPerTraj-1)*NumReading 列，这可能是个笔误。
# 循环会处理所有 NumPointPerTraj 个点，所以正确的列数应该是 NumPointPerTraj * NumReading。
# 我们在这里使用正确的尺寸。
RNN_Database = np.zeros((L, NumPointPerTraj * NumReading))

print("Processing trajectories to create RNN input...")
for ii in range(L):  # 遍历每一条轨迹
    # 打印进度
    if (ii + 1) % 1000 == 0:
        print(f"  Processing trajectory {ii + 1}/{L}")

    # 用于在 RNN_Database 中定位列的计数器
    cnt_col_rnn = 0

    for jj in range(NumPointPerTraj):  # 遍历轨迹中的每一个点

        # 获取当前点的 (X, Y) 坐标
        # MATLAB: TrajData(ii,(jj-1)*2+1) -> Python: TrajData[ii, jj*2]
        X = TrajData[ii, jj * 2]
        Y = TrajData[ii, jj * 2 + 1]

        # 在主数据库中搜索匹配的坐标
        # MATLAB 的 for kk = 1:5:L_Data 假设每个位置的数据是分块存储的
        for kk in range(0, L_Data, 5):
            db_X = Database[kk, 0]
            db_Y = Database[kk, 1]

            # 使用 np.isclose 进行浮点数比较，比 abs(a-b)<epsilon 更稳健
            if np.isclose(db_X, X) and np.isclose(db_Y, Y):

                # 为该位置的多个 RSSI 读数随机选择一个
                # MATLAB: round(100*rand(1)) -> Python: np.random.randint(0, 101)
                rand_num = np.random.randint(0, 101)

                # 默认的目标行索引是当前匹配的行 kk
                target_row_idx = kk

                # 检查随机选择的行是否有效
                potential_idx = kk + rand_num
                if potential_idx < L_Data:
                    # 检查随机选择的行的坐标是否仍然与当前点匹配
                    rand_X = Database[potential_idx, 0]
                    rand_Y = Database[potential_idx, 1]
                    if np.isclose(rand_X, X) and np.isclose(rand_Y, Y):
                        # 如果有效，则更新目标行索引
                        target_row_idx = potential_idx

                # 提取 RSSI 数据并进行归一化
                # 使用数组切片代替 for 循环，效率更高
                # MATLAB: Database(..., 3:end) -> Python: Database[..., 2:2+NumReading]
                rssi_values = Database[target_row_idx, 2: 2 + NumReading]
                normalized_rssi = (rssi_values + 100) / 100

                # 将归一化后的 RSSI 值放入 RNN_Database 的相应位置
                start_col = cnt_col_rnn
                end_col = cnt_col_rnn + NumReading
                RNN_Database[ii, start_col:end_col] = normalized_rssi

                # 更新列计数器，为下一个点做准备
                cnt_col_rnn += NumReading

                # 找到匹配项后，跳出内部的 kk 循环，继续处理轨迹中的下一个点
                break

# --- Save the final database ---
output_filename = 'InputRNN_10points_RSSI.csv'
print(f"Saving RNN input database to {output_filename}...")
np.savetxt(output_filename, RNN_Database, delimiter=',')

print("Done.")