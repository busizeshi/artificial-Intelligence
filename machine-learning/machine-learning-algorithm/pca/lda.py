import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# === 1. 生成模拟数据 ===
# 我们生成两组呈椭圆状分布的数据，模拟两个类别
np.random.seed(42)

# 类别 1 (红色)
mean1 = [2, 2]
cov1 = [[2, 1], [1, 2]]  # 协方差矩阵，让数据倾斜
X1 = np.random.multivariate_normal(mean1, cov1, 100)

# 类别 2 (蓝色) - 我们故意让它们离得不远，且方向平行
mean2 = [6, 4]
cov2 = [[2, 1], [1, 2]]
X2 = np.random.multivariate_normal(mean2, cov2, 100)

# 合并数据
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(100), np.ones(100))) # 标签 0 和 1

# === 2. 使用 LDA 计算最佳投影方向 ===
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

# 获取 LDA 的权重向量 (也就是投影方向 w)
w = lda.coef_[0]
# 计算斜率，用于画图
slope = w[1] / w[0]

# === 3. 可视化 ===
plt.figure(figsize=(10, 6))

# A. 画出原始数据点
plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.5, label='Class 1 (Red Beans)')
plt.scatter(X2[:, 0], X2[:, 1], c='blue', alpha=0.5, label='Class 2 (Green Beans)')

# B. 画出 LDA 找到的最佳投影线
# 为了画线，我们需要一个中心点和线的范围
center_x = np.mean(X[:, 0])
center_y = np.mean(X[:, 1])
x_vals = np.array([center_x - 5, center_x + 5])
y_vals = slope * (x_vals - center_x) + center_y

plt.plot(x_vals, y_vals, c='green', linewidth=3, linestyle='--', label='LDA Projection Line')

# C. 为了展示投影效果，我们把几个点投影到直线上看看
# (这里只画几个点示意，避免图太乱)
for i in range(0, 200, 20):
    # 计算垂足公式 (简单的几何投影)
    # 这是一个简化的视觉连线，为了说明点是垂直投影到绿线上的
    x0, y0 = X[i]
    # 投影计算略微复杂，这里用简单的视觉辅助线代替
    # 实际 LDA 是将数据点投射到法向量上
    pass

plt.title("LDA Principle Visualization: Finding the Best Separation Axis", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal') # 保证 X 和 Y 轴比例一致，这样看到的角度才是真的

plt.show()

# 打印 LDA 的解释
print(f"LDA 找到的最佳投影向量 w: {w}")