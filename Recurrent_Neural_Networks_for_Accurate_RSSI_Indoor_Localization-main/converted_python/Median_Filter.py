import numpy as np


def median_filter(RSSI_Before, n):
    if n < len(RSSI_Before):  # if don't have enough samples
        RSSI_After = RSSI_Before[n]
    else:  # have enough sample
        RSSI_After = np.median(RSSI_Before)

    return RSSI_After