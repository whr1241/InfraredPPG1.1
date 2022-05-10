# 实验NeuroKit信号处理包 
# https://neurokit2.readthedocs.io/en/latest/introduction.html#photoplethysmography-ppg-bvp
import numpy as np
import pandas as pd
import neurokit2 as nk

# # Generate synthetic signals
# ecg = nk.ecg_simulate(duration=10, heart_rate=70)
# ppg = nk.ppg_simulate(duration=10, heart_rate=70)
# rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
# eda = nk.eda_simulate(duration=10, scr_number=3)
# emg = nk.emg_simulate(duration=10, burst_number=2)

# # Visualise biosignals
# data = pd.DataFrame({"ECG": ecg,
#                      "PPG": ppg,
#                      "RSP": rsp,
#                      "EDA": eda,
#                      "EMG": emg})
# nk.signal_plot(data, subplots=True)



# Generate 15 seconds of PPG signal (recorded at 250 samples / second)
ppg = nk.ppg_simulate(duration=15, sampling_rate=250, heart_rate=70)
# Process it
signals, info = nk.ppg_process(ppg, sampling_rate=250)
# Visualize the processing
nk.ppg_plot(signals, sampling_rate=250)