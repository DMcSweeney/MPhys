import itertools
import numpy as np
import random
import time
#from numba import jit
import pandas as pd

print("Load ")
hamming_set = pd.read_csv("hamming_min_mean_1000.txt", sep=",", header=None)

data_set = {}

for i in range(len(hamming_set)):
    new_set = hamming_set[0:i]
    a = np.mean(new_set)
    print(a)
    data_set.update({i:np.mean(new_set)})

#for (key,value) in data_set.items() :
    #print(key, " ::", value , "\n")
