import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt


df = pd.read_csv('Raw_ECG/0000/Ecgsubject_0000_task1_0-1_timeadjusted.csv')
raw = df["ecg"]

# Sampling rate of polar in Hz
_SAMPLING_RATE = 130

# Sample the data every _SAMPLING_STEP = n steps
_SAMPLING_STEP = 5

# Sampling Rate is 130Hz >> 30 Sec = chunksize of 7800
chunksize = 5000


def chnk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst)-n, _SAMPLING_STEP):
        print(f'Chunking raw data ... {(i/len(lst))*100}%')
        yield lst[i:i + n]
        
chnks = chnk(raw, chunksize)

newarr = []

for data in chnks:
    newarr.append(data)
    
hrv_data = pd.DataFrame()
    
for index, data in enumerate(newarr):
    print(f'Calculating HRV features ... {(index/len(newarr))*100}%')
    clean = nk.ecg_clean(data, sampling_rate=130)
    peaks = nk.ecg_peaks(clean, sampling_rate=130)
    hrv = nk.hrv_frequency(peaks[1], sampling_rate=130)
    hrv["time"] = (index*_SAMPLING_STEP) * 1/_SAMPLING_RATE
    hrv_data = pd.concat([hrv_data, hrv])


def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Apply moving average filter to specific columns (e.g., RMSSD)
hrv_data["HRV_LFHF"] = smooth_data(hrv_data["HRV_LFHF"], window_size=90)
print(hrv_data.columns)



hrv_data.to_csv("ECG_0000_Man.csv", index=False)





    
    
    







    
