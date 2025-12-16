# numpy概述
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)    # [1 2 3 4 5]
print(type(a))  # <class 'numpy.ndarray'>

a += 1
print(a)    # [2 3 4 5 6]

b=a+2
print(b)    # [4 5 6 7 8]
print(a+b)  # [ 6  8 10 12 14]

print(a[1]) # 3

print(a.shape)  # (5,)

