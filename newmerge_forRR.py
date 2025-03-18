import pandas as pd

ecg_file = "Raw_ECG/0001/Ecgsubject_0001_task2_2-1_timeadjusted.csv"
unity_file = "Unity_edit_time/0001/subject_0001_task2_2_timeadjusted.csv"

df_ecg = pd.read_csv(ecg_file)
df_unity = pd.read_csv(unity_file)
df_merged = pd.merge(df_ecg, df_unity, on='norm_time', how='outer')
df_merged = df_merged.sort_values(by='norm_time')
one_minute_threshold = df_merged["norm_time"].max() - 60

df_merged = df_merged[df_merged["norm_time"] <= one_minute_threshold]

df_merged["Character_status"] = df_merged["Character_status"].fillna(method="ffill").fillna("Walk")
columns_order = ["norm_time", "Character_status", "ecg", "hr", "rr", "marker"]
df_final = df_merged[columns_order]
df_final.to_csv("0001_woman.csv", index=False)

