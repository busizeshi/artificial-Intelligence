import numpy as np

array1=np.array([1,2,3],dtype=np.float32)
print(array1)
# [1. 2. 3.]
print(array1.nbytes)
# 12

array2=np.array([1,2.0,"哈士奇"],dtype=object)
print(array2)
# [1 2.0 '哈士奇']
print(array2*2)
# [2. 4.0 '哈士奇哈士奇']
np.asarray(array2,dtype=str)
print(array2)
# [1 2.0 '哈士奇']
array2.astype( object)
print(array2)
# [1 2.0 '哈士奇']