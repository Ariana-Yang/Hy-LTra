def average_filter(RSSI_before, F1_before, F2_before, n):
    beta1 = 0.2
    beta2 = 0.8
    beta3 = 0.05
    beta4 = 0.15
    beta5 = 0.8

    time_count = n
    F1 = F1_before.copy()
    F2 = F2_before.copy()

    if n == 1:
        F1[0] = RSSI_before[0]
        F2[0] = RSSI_before[0]
        RSSI_after = RSSI_before[0]

    if n == 2:
        F1[1] = beta1 * RSSI_before[n - 2] + beta2 * RSSI_before[n - 1]
        F2[1] = beta1 * F1[n - 2] + beta2 * F1[n - 1]
        F3 = beta1 * F2[n - 2] + beta2 * F2[n - 1]
        RSSI_after = F3

    if n >= 3:
        if n > 3:
            n = 3
            F1[0], F1[1] = F1[1], F1[2]
            F2[0], F2[1] = F2[1], F2[2]
        F1[2] = beta3 * RSSI_before[n - 3] + beta4 * RSSI_before[n - 2] + beta5 * RSSI_before[n - 1]
        F2[2] = beta3 * F1[n - 3] + beta4 * F1[n - 2] + beta5 * F1[n - 1]
        F3 = beta3 * F2[n - 3] + beta4 * F2[n - 2] + beta5 * F2[n - 1]
        RSSI_after = F3

    time_count += 1
    return RSSI_after, F1, F2, time_count