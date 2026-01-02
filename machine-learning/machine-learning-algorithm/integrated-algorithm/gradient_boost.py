import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 1. 生成带噪声的非线性数据
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

# 准备绘图数据
X_new = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)

def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    # 累加所有弱学习器的预测
    y_pred = sum(regressor.predict(X_new) for regressor in regressors)
    plt.plot(X, y, data_style, label=data_label)
    plt.plot(X_new, y_pred, style, linewidth=2, label=label)
    plt.axis(axes)

# 2. 模拟梯度提升的过程（手动模拟 3 棵树的累加）
# 树 1: 拟合原始数据
gbrt1 = GradientBoostingRegressor(max_depth=2, n_estimators=1, learning_rate=1.0, random_state=42)
gbrt1.fit(X, y)

# 树 2: 拟合第一棵树的残差
y2 = y - gbrt1.predict(X)
gbrt2 = GradientBoostingRegressor(max_depth=2, n_estimators=1, learning_rate=1.0, random_state=42)
gbrt2.fit(X, y2)

# 树 3: 拟合前两棵树累加后的残差
y3 = y2 - gbrt2.predict(X)
gbrt3 = GradientBoostingRegressor(max_depth=2, n_estimators=1, learning_rate=10, random_state=42)
gbrt3.fit(X, y3)

# 3. 绘图对比
plt.figure(figsize=(15, 11))

# 左侧：每一棵单独的树在拟合什么
plt.subplot(3, 2, 1)
plot_predictions([gbrt1], X, y, [-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", data_label="Training set")
plt.title("Residuals and tree predictions")

plt.subplot(3, 2, 3)
plot_predictions([gbrt2], X, y2, [-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", data_style="g+", data_label="Residuals")

plt.subplot(3, 2, 5)
plot_predictions([gbrt3], X, y3, [-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", data_style="g+")

# 右侧：集成模型（累加）后的预测效果
plt.subplot(3, 2, 2)
plot_predictions([gbrt1], X, y, [-0.5, 0.5, -0.1, 0.8], label="$F_1(x_1) = h_1(x_1)$")
plt.title("Ensemble predictions")

plt.subplot(3, 2, 4)
plot_predictions([gbrt1, gbrt2], X, y, [-0.5, 0.5, -0.1, 0.8], label="$F_2(x_1) = h_1+h_2$")

plt.subplot(3, 2, 6)
plot_predictions([gbrt1, gbrt2, gbrt3], X, y, [-0.5, 0.5, -0.1, 0.8], label="$F_3(x_1) = h_1+h_2+h_3$")

plt.tight_layout()
plt.show()