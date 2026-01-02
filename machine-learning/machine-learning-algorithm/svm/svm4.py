"""
非线性支持向量机
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. 生成非线性数据 (月亮形数据)
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 标准化 (对含有核函数的SVM尤其重要)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 定义三个不同核函数的SVM模型
# 线性核 (注定失败)
svm_linear = SVC(kernel="linear", C=10)
# 多项式核 (3阶多项式)
svm_poly = SVC(kernel="poly", degree=3, coef0=1, C=5)
# 高斯 RBF 核 (最强大)
svm_rbf = SVC(kernel="rbf", gamma=0.5, C=1) # gamma稍后解释

# 训练模型
svm_linear.fit(X_scaled, y)
svm_poly.fit(X_scaled, y)
svm_rbf.fit(X_scaled, y)

# --- 绘图辅助函数 (用于画出非线性的决策边界/等高线) ---
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X_new).reshape(x0.shape)
    y_decision = clf.decision_function(X_new).reshape(x0.shape)
    # 绘制决策边界和间隔区域的等高线
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

# --- 可视化对比 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# 图1: 线性核
plt.sca(axes[0])
plot_predictions(svm_linear, [-2.5, 2.5, -2, 2])
plt.plot(X_scaled[:, 0][y==0], X_scaled[:, 1][y==0], "bs")
plt.plot(X_scaled[:, 0][y==1], X_scaled[:, 1][y==1], "g^")
plt.title("Linear Kernel (Failed)", fontsize=14)

# 图2: 多项式核
plt.sca(axes[1])
plot_predictions(svm_poly, [-2.5, 2.5, -2, 2])
plt.plot(X_scaled[:, 0][y==0], X_scaled[:, 1][y==0], "bs")
plt.plot(X_scaled[:, 0][y==1], X_scaled[:, 1][y==1], "g^")
plt.title("Polynomial Kernel (degree=3)", fontsize=14)

# 图3: RBF核
plt.sca(axes[2])
plot_predictions(svm_rbf, [-2.5, 2.5, -2, 2])
plt.plot(X_scaled[:, 0][y==0], X_scaled[:, 1][y==0], "bs")
plt.plot(X_scaled[:, 0][y==1], X_scaled[:, 1][y==1], "g^")
plt.title("RBF / Gaussian Kernel", fontsize=14)

plt.show()