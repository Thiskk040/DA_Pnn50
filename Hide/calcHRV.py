import pandas as pd
import matplotlib.pyplot as plt

# ตัวอย่างข้อมูล
data = pd.DataFrame({
    'timestamp': ['2024-01-01 00:00:01', '2024-01-01 00:00:02', '2024-01-01 00:00:03', '2024-01-01 00:00:04',
                  '2024-01-02 00:00:01', '2024-01-02 00:00:02', '2024-01-02 00:00:03', '2024-01-02 00:00:04'],
    'lf': [0.45, 0.47, 0.46, 0.48, 0.50, 0.49, 0.51, 0.52],
    'hf': [0.55, 0.53, 0.54, 0.52, 0.50, 0.51, 0.49, 0.48],
    'event': ['start', 'playing', 'playing', 'end', 'start', 'playing', 'playing', 'end'],
    'position': [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
    'response': ['correct', 'wrong', 'correct', 'correct', 'wrong', 'correct', 'wrong', 'correct']
})

# แปลง timestamp เป็น datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Section 1: 2024-01-01
section1 = data[(data['timestamp'] >= '2024-01-01') & (data['timestamp'] < '2024-01-02')]
print(section1)

# Section 2: 2024-01-02
section2 = data[(data['timestamp'] >= '2024-01-02') & (data['timestamp'] < '2024-01-03')]

# Plot Section 1
plt.figure(figsize=(12, 6))
plt.plot(section1['timestamp'], section1['lf'], label='LF')
plt.plot(section1['timestamp'], section1['hf'], label='HF')

# Annotate event, position, and response
for i, row in section1.iterrows():
    plt.annotate(f"Event: {row['event']}\nPos: {row['position']}\nResp: {row['response']}",
                 (row['timestamp'], row['lf']),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')

plt.xlabel('Time')
plt.ylabel('LF/HF')
plt.title('LF/HF Time Series on 2024-01-01')
plt.legend()
plt.show()

# Plot Section 2
plt.figure(figsize=(12, 6))
plt.plot(section2['timestamp'], section2['lf'], label='LF')
plt.plot(section2['timestamp'], section2['hf'], label='HF')

# Annotate event, position, and response
for i, row in section2.iterrows():
    plt.annotate(f"Event: {row['event']}\nPos: {row['position']}\nResp: {row['response']}",
                 (row['timestamp'], row['lf']),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')

plt.xlabel('Time')
plt.ylabel('LF/HF')
plt.title('LF/HF Time Series on 2024-01-02')
plt.legend()
plt.show()


