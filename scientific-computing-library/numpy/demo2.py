# array结构
import numpy as np

list1 = [1, 2, 3]
a = np.array(list1)
print(a)  # [1 2 3]

type(a)
print(a.dtype)  # int64

# 对于ndarray结构来说，数据类型会根据输入数据进行转换。会向下转换
list2 = [1, 2, 3.0]
b = np.array(list2)
print(b)  # [1. 2. 3.]
print(b.dtype)  # float64

print(a.itemsize)  # 8

print(a.size)  # 3

print(np.size(a))  # 3
print(np.shape(a))  # (3,)

print(a.ndim)  # 1

a.fill(0)
print(a)  # [0 0 0]

# 索引与切片
# 和python一致
a = np.array([1, 2, 3, 4, 5])
print(a[0])  # 1
print(a[-1])  # 5
print(a[1:4])  # [2 3 4]

# 矩阵格式
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)  # [[1 2 3]
#  [4 5 6]]
print(a.shape)  # (2, 3)
print(a.ndim)  # 2
print(a.size)  # 6
print(a.itemsize)  # 8

print(a[0, 0])  # 1
a[0, 0] = 100
print(a)  # [[100   2   3]
#  [  4   5   6]]
print(a[1])  # [4 5 6]
print(a[:, 0])  # [100   4]

print(a[0, 0:2])  # [100   2]

b = np.array([[1, 2, 3], [4, 5, 6]])
c = b
# 浅拷贝
c[0, 0] = 100
print(b)  # [[100   2   3]
#  [  4   5   6]]

c = b.copy()
c[0, 0] = 200
print(b)  # [[100   2   3]
#   [  4   5   6]]

a = np.arange(0, 10, 2)
print(a)  # [0 2 4 6 8]
a = np.arange(0, 10)
print(a)  # [0 1 2 3 4 5 6 7 8 9]

mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
a = np.arange(0, 10)
print(mask)  # [ True False  True False  True False  True False  True False]
print(a[mask])  # [0 2 4 6 8]
print(a[mask == False])  # [1 3 5 7 9]

random_array = np.random.randint(0, 10, (3, 3))
print(random_array)
# [[3 8 1]
#  [2 0 0]
#  [7 3 6]]
mask = random_array > 5
print(mask)  # [ True  True  True]]
#  [ True  True  True]]
#    [ True False  True]]
