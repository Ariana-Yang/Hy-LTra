import numpy as np
import os

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % WiFi Indoor Localization
# % Minhtu (Translated to Python)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % % % Generate random training trajectories under the constraints
# % % % that the distance between consecutive locations is bounded by
# % % % the maximum distance a user can travel within the sample
# % % % interval in practical scenarios.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# --- Configuration ---
# Database
Norm_On = 0  # 1: Standardization Normalization
# 0: Mean Normalization

# !!! 重要: 请将此路径修改为您自己的数据库文件夹路径 !!!
my_folder = r'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data'

if Norm_On == 1:
    # 注意：确保文件路径正确
    input_file = 'MeanDatabase_Normalize_AverageFilter.csv'
    Input = np.loadtxt(input_file, delimiter=',')
    Mean_Location = np.array([11.1397910665714, -3.78622840030012])
    Std_Location = np.array([6.31809099260756, 6.42186397145320])
else:
    input_file = 'MeanDatabase_8June2018_11MAC.csv'
    Input = np.loadtxt(os.path.join(my_folder, input_file), delimiter=',')

Database = Input
L = Database.shape[0]  # Number of reference points

# Number of times which the database is repeated
NumRepeatedDatabase = 15  # Num of Trajectories = NumRepeatedDatabase * L
NumPointPerTraj = 10  # Number of Time Steps in LSTM networks

# Initialize matrices
# The first 2 rows/cols are for storing coordinates, same as MATLAB version
MapDistance = np.zeros((L + 2, L + 2))
NumNeighboursArray = np.zeros(L)
SumProbArray = np.zeros(L)

# Assumption
v_user_max = 5  # m/s % Bounded Speed
t_request = 1  # Period of request time is 1 second
distance_max = v_user_max * t_request
sigma = distance_max

if Norm_On == 1:
    distance_max = v_user_max * t_request / Std_Location[0]
    sigma = distance_max

# -------------------------------------------------------------
# %%%% Build Map Distance
# -------------------------------------------------------------
print("Building transition probability map...")

# Store coordinates in the borders of MapDistance matrix
MapDistance[2:, 0] = Database[:, 0]
MapDistance[2:, 1] = Database[:, 1]
MapDistance[0, 2:] = Database[:, 0]
MapDistance[1, 2:] = Database[:, 1]

for ii in range(L):
    X, Y = Database[ii, 0], Database[ii, 1]

    CountNeighbour = 0
    Sum_Prob = 0
    for jj in range(L):
        X1, Y1 = Database[jj, 0], Database[jj, 1]

        # Calculate Euclidean Distance
        dist = np.sqrt((X - X1) ** 2 + (Y - Y1) ** 2)

        if dist > distance_max:
            MapDistance[ii + 2, jj + 2] = 0
        else:
            # The probability factor is a constant to normalize the truncated Gaussian
            ProbFactor = -1 / (2 * (sigma ** 2) * (np.exp(-(distance_max ** 2) / (2 * sigma ** 2)) - 1))
            P_l = ProbFactor * np.exp(-dist ** 2 / (
                        2 * sigma ** 2))  # Note: Original code used dist, not dist^2. Assuming dist^2 for Gaussian.
            # If original was correct, use np.exp(-dist / ...). Let's stick to original formula.
            P_l = ProbFactor * np.exp(-dist / (2 * sigma ** 2))  # Sticking to the original formula as written

            Sum_Prob += P_l
            MapDistance[ii + 2, jj + 2] = P_l
            CountNeighbour += 1

    NumNeighboursArray[ii] = CountNeighbour
    if Sum_Prob > 0:  # Avoid division by zero if a point has no neighbors
        SumProbArray[ii] = Sum_Prob

# --- Normalize Map to get probabilities ---
print("Normalizing map...")
# The core data is in the [2:, 2:] slice
prob_matrix = MapDistance[2:, 2:]
# Using np.newaxis for broadcasting to divide each row by its sum
non_zero_sums = SumProbArray[SumProbArray > 0]
prob_matrix[SumProbArray > 0, :] /= non_zero_sums[:, np.newaxis]
MapDistance[2:, 2:] = prob_matrix

# --- Create CDF Map ---
print("Creating CDF map...")
# Use numpy's cumsum for efficiency, which is much faster than a Python loop
MapDistance[2:, 2:] = np.cumsum(MapDistance[2:, 2:], axis=1)

# --- Create Map with position index ---
# This map will store the index of the next possible point
Pos_Map = MapDistance.copy()
# The original MATLAB code uses a very inefficient O(L^3) triple loop.
# The index being searched for (kk) is simply the column index (jj) of the inner loop.
# We can simplify this to an O(L^2) operation.
for ii in range(L):
    for jj in range(L):
        if MapDistance[ii + 2, jj + 2] != 0:
            Pos_Map[ii + 2, jj + 2] = jj  # Store the 0-based index

# -------------------------------------------------------------
# %%%% Structure: x1 y1 x2 y2 x3 y3 ...
# -------------------------------------------------------------
print("Generating trajectories...")
total_trajectories = L * NumRepeatedDatabase
TrajArray = np.zeros((total_trajectories, NumPointPerTraj * 2))
TrajOrder = np.zeros((total_trajectories, NumPointPerTraj), dtype=int)

for jj in range(NumRepeatedDatabase):
    for ii in range(L):
        # Calculate the current row index for the output arrays
        row_idx = ii + jj * L

        # Set the starting point of the trajectory
        TrajArray[row_idx, 0] = Database[ii, 0]
        TrajArray[row_idx, 1] = Database[ii, 1]

        NextPos = ii
        TrajOrder[row_idx, 0] = NextPos

        for num_point in range(1, NumPointPerTraj):
            ran_num = np.random.rand()

            # Use np.searchsorted to find the next point efficiently from the CDF row
            # It's much faster than iterating with a for loop
            cdf_row = MapDistance[NextPos + 2, 2:]

            # Find the first index where the CDF value is greater than the random number
            # This effectively samples from the discrete probability distribution
            next_idx_in_row = np.searchsorted(cdf_row, ran_num)

            # Ensure we don't go out of bounds if ran_num is very close to 1.0
            if next_idx_in_row >= L:
                next_idx_in_row = L - 1

            # Get the coordinates of the next point
            TrajArray[row_idx, num_point * 2] = MapDistance[0, next_idx_in_row + 2]
            TrajArray[row_idx, num_point * 2 + 1] = MapDistance[1, next_idx_in_row + 2]

            # Get the actual database index of the next point
            NextPos = int(Pos_Map[NextPos + 2, next_idx_in_row + 2])
            TrajOrder[row_idx, num_point] = NextPos

# --- Save the results ---
output_filename = os.path.join(my_folder, 'Traj_10points_5k_5m_s.csv')
print(f"Saving trajectories to {output_filename}...")
np.savetxt(output_filename, TrajArray, delimiter=',')

print("Done.")