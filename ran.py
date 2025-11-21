import random
from collections import Counter
numbers = [random.choice([51, 54]) for _ in range(10)]
print("Numbers picked 10 times:", numbers)

count = Counter(numbers)

most_common_number, freq = count.most_common(1)[0]
print(f"The most frequent number: {most_common_number} (appeared {freq} times)")
