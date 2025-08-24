import numpy as np
import pandas as pd

# Parameter - 1 Unit = 40 inches
Num_Mac = 11  # Number of APs per vector in database

# Normalize ------------------------------------
# r = (r - mean) / sigma
# -----------------------------------------------

# Test
myFolder = r'C:\Users\minh_\Desktop\CSI_RSSI_Database\RSSI_6AP_Experiments\Nexus4_Data\'  # Database Folder
InputTest = pd.read_csv(myFolder + 'UpdatedDatabase_8June2018_11MAC_AverageFilter.csv', header=None)
Database = InputTest.values
LengthDatabase = Database.shape[0]  # Number of Test points

RSSI_Array = Database[:, 2:]
Location_Array = Database[:, :2]
NumLocation = len(Location_Array)

# Locations
MeanArray_Location = np.zeros(2)
StdArray_Location = np.zeros(2)
for ii in range(2):
    MeanArray_Location[ii] = np.mean(Location_Array[:, ii])

for Count in range(NumLocation):
    for CountMac in range(2):
        StdArray_Location[CountMac] += (Location_Array[Count, CountMac] - MeanArray_Location[CountMac]) ** 2

StdArray_Location = StdArray_Location / LengthDatabase
StdArray_Location = np.sqrt(StdArray_Location)

# RSSI
# Calculate Mean & Standard deviation for every AP
MeanArray_RSSI = np.zeros(Num_Mac)

# Mean Calculation
for Count in range(LengthDatabase):
    for CountMac in range(Num_Mac):
        MeanArray_RSSI[CountMac] += RSSI_Array[Count, CountMac]

MeanArray_RSSI = np.round(MeanArray_RSSI / LengthDatabase)

# Standard Deviation
StdArray_RSSI = np.zeros(Num_Mac)
for Count in range(LengthDatabase):
    for CountMac in range(Num_Mac):
        StdArray_RSSI[CountMac] += (RSSI_Array[Count, CountMac] - MeanArray_RSSI[CountMac]) ** 2

StdArray_RSSI = StdArray_RSSI / LengthDatabase
StdArray_RSSI = np.round(np.sqrt(StdArray_RSSI))

# Normalized Step
Normalized_Database = np.zeros_like(Database)
for Count in range(LengthDatabase):
    # Location
    for ii in range(2):
        Normalized_Database[Count, ii] = (Database[Count, ii] - MeanArray_Location[ii]) / StdArray_Location[ii]
    # RSSI
    for ii in range(3, 13):
        Normalized_Database[Count, ii] = (Database[Count, ii] - MeanArray_RSSI[ii - 2]) / StdArray_RSSI[ii - 2]

# Save normalized database
np.savetxt(myFolder + 'Normalize_Database_AverageFilter.csv', Normalized_Database, delimiter=',')