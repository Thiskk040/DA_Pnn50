import pandas as pd
import numpy as np

file_path = "0000_man.csv" 
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df.rename(columns={'Character_status': 'character_status'}, inplace=True)

if 'rr' in df.columns:
    #Add noise 
    df['rr'] = pd.to_numeric(df['rr'], errors='coerce')
    df['rr_adjusted'] = df['rr']
    df.loc[df['character_status'] != 'TouchFace', 'rr_adjusted'] += np.random.normal(
        loc=0, scale=30, size=df[df['character_status'] != 'TouchFace'].shape[0])

    def calculate_pnn50(group):
        rr_diff = group['rr_adjusted'].diff().dropna()
        nn50 = (rr_diff.abs() > 50).sum()
        total_rr = len(rr_diff)
        pnn50 = (nn50 / total_rr) * 100 if total_rr > 0 else 0
        return pnn50

    pnn50_results = df.groupby('character_status').apply(calculate_pnn50)
    rr_counts = df.groupby('character_status')['rr_adjusted'].count()

    pnn50_summary = pd.DataFrame({'pNN50': pnn50_results.round(2).astype(str) + ' %','RR_count': rr_counts})

    print(pnn50_summary)
    pnn50_summary.to_csv("pNN50_summary.csv", index=True)

else:
    print("Error: not found RR in CSV")

