import numpy as np
import pandas as pd

FILTER_OPTION = 1  # 1: Average Filter, 0: Median Filter

Num_Mac = 11  # Number of APs per vector in database

# Test
myFolder = r'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data\'  # Database Folder
Database = pd.read_csv(myFolder + 'UpdatedDatabase_24Jan2018_Full.csv').values
LengthDatabase = Database.shape[0]  # Number of RPs

# Split location from Database
PreX = Database[0, 0]
PreY = Database[0, 1]
StartingPoint = 0
CountLocation = 0

for CountBlock in range(1, LengthDatabase):
    X = Database[CountBlock, 0]
    Y = Database[CountBlock, 1]
    if (X != PreX) or (Y != PreY) or (CountBlock == LengthDatabase - 1):
        PreX = X
        PreY = Y
        EndingPoint = CountBlock - 1
        CountLocation += 1  # Count Number of Location
        LengthBlock = EndingPoint - StartingPoint + 1

        # Filter RSSI in a specific location
        RSSI_Original = Database[StartingPoint:EndingPoint + 1, 2:]
        RSSI_Filtered = np.zeros_like(RSSI_Original)
        F1_Before = np.zeros(3)
        F2_Before = np.zeros(3)
        RSSI_Before = np.zeros(3)

        for CountMac in range(Num_Mac):
            RSSI_Array_Temp = RSSI_Original[:, CountMac]
            RSSI_Array_After = np.zeros(LengthBlock)
            MeanValue = np.mean(RSSI_Array_Temp)
            n = 0

            for CountPoint in range(LengthBlock):
                RSSI_Temp = RSSI_Array_Temp[CountPoint]
                if RSSI_Temp == -100:  # Avoid -100
                    RSSI_Temp = MeanValue
                if n < 3:
                    RSSI_Before[n] = RSSI_Temp
                else:
                    RSSI_Before[0] = RSSI_Before[1]
                    RSSI_Before[1] = RSSI_Before[2]
                    RSSI_Before[2] = RSSI_Temp

                # Median Filter
                if FILTER_OPTION == 0:  # Median Filter
                    RSSI_After_Median = Median_Filter(RSSI_Before, n + 1)
                    RSSI_Array_After[CountPoint] = round(RSSI_After_Median)
                    n += 1
                else:
                    # Average Filter
                    RSSI_After, F1, F2, TimeCount = Average_Filter(RSSI_Before, F1_Before, F2_Before, n + 1)
                    n = TimeCount
                    F1_Before = F1
                    F2_Before = F2
                    RSSI_Array_After[CountPoint] = round(RSSI_After)

            RSSI_Filtered[:, CountMac] = RSSI_Array_After

        # Update to Database
        Database[StartingPoint:EndingPoint + 1, 2:] = RSSI_Filtered
        StartingPoint = CountBlock  # Restart Starting Point

# Save updated database
np.savetxt(myFolder + 'UpdatedDatabase_8June2018_11MAC_AverageFilter.csv', Database, delimiter=',')