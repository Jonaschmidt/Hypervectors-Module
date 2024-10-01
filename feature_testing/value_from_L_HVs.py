'''
Demonstration of how to recover/assign values from a dictionary of L_HVs
'''

import numpy as np
import tensorflow as tf

import hypervectors as hv

hv_size = 1024
value_range = (0, 42)

min_val = value_range[0]
max_val = value_range[1]

LHVs = hv.gen_L_HVs(hv_size=hv_size, value_range=value_range, random_method="Sobol")

'''
print(np.sum(LHVs[0].tensor.numpy()))
print(np.sum(LHVs[42].tensor.numpy()))
print()
'''

for i in range(min_val, max_val + 1):
    print("Expected:", i)
    print("Actual:  ", ((np.count_nonzero(LHVs[i].tensor.numpy() == 1)) / hv_size) * max_val)
    print()

