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
NumPointPerTraj = SizeTraj[1] // 2  # How many Time Steps in LSTM trajectory

print(f"Number of trajectories: {L}")
print(f"Points per trajectory: {NumPointPerTraj}")

# -----------------------------------------------------------
# %%%% Add RSSI to the trajectory
# %%%% Format: [loc_1, rssi_2, loc_2, rssi_3, ..., loc_{N-1}, rssi_N]
# -----------------------------------------------------------

# 初始化用于存储 RNN 输入的数据库
# 循环将运行 NumPointPerTraj - 1 次。
# 每次循环，我们会添加 2 个坐标值和 NumReading 个 RSSI 值。
# 因此，总列数应为 (NumPointPerTraj - 1) * (2 + NumReading)。
# 注意：这与原始 MATLAB 代码的初始化大小不同，但与其实际的循环逻辑相符。
num_cols = (NumPointPerTraj - 1) * (2 + NumReading)
RNN_Database = np.zeros((L, num_cols))

print("Processing trajectories to create interleaved Location-RSSI input...")
for ii in range(L):  # 遍历每一条轨迹
    # 打印进度
    if (ii + 1) % 1000 == 0:
        print(f"  Processing trajectory {ii + 1}/{L}")

    # 用于在 RNN_Database 中定位列的计数器
    cnt_col_rnn = 0

    # 循环 N-1 次，因为我们总是用当前位置和 *下一个* 位置的 RSSI
    # MATLAB: for jj = 1:NumPointPerTraj with an if break -> Python: range(NumPointPerTraj - 1)
    for jj in range(NumPointPerTraj - 1):

        # 1. 获取并存储当前点 (jj) 的坐标
        current_X = TrajData[ii, jj * 2]
        current_Y = TrajData[ii, jj * 2 + 1]

        RNN_Database[ii, cnt_col_rnn] = current_X
        RNN_Database[ii, cnt_col_rnn + 1] = current_Y
        cnt_col_rnn += 2

        # 2. 获取下一个点 (jj+1) 的坐标，以查找其对应的 RSSI
        next_X = TrajData[ii, (jj + 1) * 2]
        next_Y = TrajData[ii, (jj + 1) * 2 + 1]

        # 3. 在主数据库中搜索下一个点的坐标以获取 RSSI
        found_match = False
        for kk in range(0, L_Data, 5):
            db_X = Database[kk, 0]
            db_Y = Database[kk, 1]

            # 使用 np.isclose 进行浮点数比较
            if np.isclose(db_X, next_X) and np.isclose(db_Y, next_Y):

                # 随机选择一个与该位置关联的 RSSI 读数
                rand_num = np.random.randint(0, 101)

                target_row_idx = kk
                potential_idx = kk + rand_num
                if potential_idx < L_Data:
                    if np.isclose(Database[potential_idx, 0], next_X) and np.isclose(Database[potential_idx, 1],
                                                                                     next_Y):
                        target_row_idx = potential_idx

                # 提取 RSSI 数据并进行归一化
                rssi_values = Database[target_row_idx, 2: 2 + NumReading]
                normalized_rssi = (rssi_values + 100) / 100

                # 将归一化后的 RSSI 值放入 RNN_Database
                start_col = cnt_col_rnn
                end_col = cnt_col_rnn + NumReading
                RNN_Database[ii, start_col:end_col] = normalized_rssi

                # 更新列计数器
                cnt_col_rnn += NumReading

                # 标记已找到并跳出搜索循环
                found_match = True
                break

        # 如果在数据库中找不到匹配的坐标（不太可能发生），则打印警告
        if not found_match:
            print(f"Warning: No match found for coordinates ({next_X}, {next_Y}) in trajectory {ii}")

# --- Save the final database ---
output_filename = os.path.join(my_folder, 'Input_Location_RSSI_10points.csv')
print(f"Saving combined database to {output_filename}...")
np.savetxt(output_filename, RNN_Database, delimiter=',')

print("Done.")