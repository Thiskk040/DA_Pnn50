import pandas as pd
import os 

fns = os.listdir(r'ConvertToBaseline')

for fn in fns:
    path = os.path.join(r'ConvertToBaseline', fn)
    df = pd.read_csv(path)
    baseline_df = df.loc[(df['time'] <= 30)][['HRV_LFHF', 'time']]
    
    newfn = "baseline_" + fn
    baseline_df.to_csv(r'Baseline/' + newfn)



