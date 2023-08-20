import torch
import numpy as np



a = np.array([True, True, False, False, False, False, False])
b = np.array([0, 2, 3, 1, 3, 4, 1])

b = b[~a]

print(b)
