import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_covariance(mean, cov, title, ax):
    # 生成数据
    x, y = np.random.multivariate_normal(mean, cov, 500).T

    # 画散点图
    ax.scatter(x, y, alpha=0.5)
    ax.set_title(title)
    ax.axis('equal')  # 保证比例一致，这样才能看清形状
    ax.grid(True)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 无相关性 (协方差 = 0) -> 圆形/正团状
mean = [0, 0]
cov_1 = [[1, 0],
         [0, 1]]
plot_covariance(mean, cov_1, "1. Covariance = 0 (No Correlation)\nRound Cloud", axes[0])

# 2. 正相关 (协方差 > 0) -> 向右上倾斜
cov_2 = [[1, 0.8],
         [0.8, 1]]
plot_covariance(mean, cov_2, "2. Positive Covariance\nTilted Up-Right (/)", axes[1])

# 3. 负相关 (协方差 < 0) -> 向左上倾斜
cov_3 = [[1, -0.8],
         [-0.8, 1]]
plot_covariance(mean, cov_3, "3. Negative Covariance\nTilted Up-Left (\)", axes[2])

plt.show()

# 打印一下具体的矩阵看看
print("矩阵 2 (正相关):")
print(np.array(cov_2))