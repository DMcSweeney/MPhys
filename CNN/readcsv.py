import pandas as pd

trainIndex = pd.read_csv("mixed_hamming_set.txt", sep=",", header=None)

trainIndex.to_csv("mixedham.csv", sep=',', encoding='utf-8')
