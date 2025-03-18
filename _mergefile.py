import pandas as pd

unityFile_df  = pd.read_csv('Unity_edit_time/0000/Subject_0000_task1_0_timeadjusted.csv') 
lfhfFile_df = pd.read_csv('ECG_0000_Man.csv')

unityFile_df['time_rounded'] = unityFile_df['norm_time'].round(1)
#lfhfFile_df['norm_time_rounded'] = lfhfFile_df['time'].round(3)

merged_df = pd.merge(unityFile_df, lfhfFile_df, left_on='time_rounded', right_on='time')
merged_df.to_csv('ECG_Merged_0000_Man.csv', index=False)

