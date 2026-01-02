"""
软间隔
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. 创建带有一个“噪点”的数据集
Xoutliers = np.array([[3.4, 1.3], [3.2, 0.8]]) # 故意放两个靠近中间的点
youtliers = np.array([0, 0])
X_orig = np.array([[1.5, 0.5], [2.0, 1.0], [4.5, 1.5], [5.0, 1.8]])
y_orig = np.array([0, 0, 1, 1])

X = np.concatenate([X_orig, Xoutliers])
y = np.concatenate([y_orig, youtliers])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 训练两个模型
# C=0.1 软间隔：大度，允许犯错以换取宽边界
svm_soft = SVC(kernel="linear", C=0.1).fit(X_scaled, y)
# C=100 硬间隔：死板，必须分对每一个点
svm_hard = SVC(kernel="linear", C=100).fit(X_scaled, y)

# 绘图函数
def plot_svc_decision_boundary(svm_clf, xmin, xmax, label):
    if not hasattr(svm_clf, 'coef_') or not hasattr(svm_clf, 'intercept_'):
        print(f"Warning: SVM model doesn't have coef_ or intercept_ attributes. Current label: {label}")
        return
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, decision_boundary + margin, "k--", linewidth=1)
    plt.plot(x0, decision_boundary - margin, "k--", linewidth=1)
    plt.title(label)

plt.figure(figsize=(12, 5))

# 左图：软间隔 (C=0.1)
plt.subplot(121)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', edgecolors='k')
plot_svc_decision_boundary(svm_soft, -2, 2, "Soft Margin (Low C=0.1)\nMore tolerance, Wider Margin")
plt.axis((-2, 2, -2, 2))

# 右图：硬间隔 (C=100)
plt.subplot(122)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', edgecolors='k')
plot_svc_decision_boundary(svm_hard, -2, 2, "Hard Margin (High C=100)\nLess tolerance, Narrower Margin")
plt.axis((-2, 2, -2, 2))

plt.show()