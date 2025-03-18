import matplotlib.pyplot as plt
import pandas as pd
#input flies 
df  = pd.read_csv('ECG_0021_Man.csv')
xaxis = df['time']
yaxis = df['HRV_LFHF']
plt.figure(figsize=(12, 6))
plt.title('Lf/HF Robot Avatar',fontsize = 35)
plt.xlabel('Time', fontsize=15)
plt.ylabel('LF/HF',fontsize = 15)
plt.text(60,0.5,'Subject = 0004')
plt.text(60,0.25,'Samling step = 15')
plt.plot(xaxis,yaxis)
plt.show()