"""
svm数据标准化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. 构造极端比例的数据（模拟未标准化的原始数据）
# 特征0范围很大，特征1范围很小
Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
ys = np.array([0, 0, 1, 1])

# 2. 训练未缩放的模型
svm_clf_unscaled = SVC(kernel="linear", C=100)
svm_clf_unscaled.fit(Xs, ys)

# 3. 训练缩放后的模型
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xs)
svm_clf_scaled = SVC(kernel="linear", C=100)
svm_clf_scaled.fit(X_scaled, ys)

# 绘图函数
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

# 可视化对比
plt.figure(figsize=(12, 5))

# 左图：未缩放（看起来很糟糕，间隔几乎看不见）
plt.subplot(121)
plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
plot_svc_decision_boundary(svm_clf_unscaled, 0, 6)
plt.xlabel("$x_0$", fontsize=12)
plt.ylabel("$x_1$  ", fontsize=12, rotation=0)
plt.title("Unscaled (Raw Data)", fontsize=14)
plt.axis((0, 6, 0, 100))

# 右图：已缩放（间隔宽阔且均匀）
plt.subplot(122)
plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")
plot_svc_decision_boundary(svm_clf_scaled, -2, 2)
plt.xlabel("$x_0'$ (standardized)", fontsize=12)
plt.title("Scaled (Standardized)", fontsize=14)
plt.axis((-2, 2, -2, 2))

plt.tight_layout()
plt.show()