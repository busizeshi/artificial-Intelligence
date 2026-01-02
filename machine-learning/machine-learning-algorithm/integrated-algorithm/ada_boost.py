import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 1. 准备数据：选取后两个类别 (1, 2) 和后两个特征 (Petal length, Petal width)
iris = datasets.load_iris()
X = iris.data[50:, 2:]  # 只要 Versicolor 和 Virginica
y = iris.target[50:] - 1  # 转换为 0 和 1 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_decision_boundary(clf, X, y, alpha=0.1):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=alpha, cmap=plt.cm.Paired)


# 2. 模拟 AdaBoost 迭代过程
m = len(X_train)
plt.figure(figsize=(14, 5))

# 演示两种学习率：1.0 (激进) 和 0.5 (稳健)
for subplot, learning_rate in ((121, 1.0), (122, 0.5)):
    sample_weights = np.ones(m)  # 初始化权重全为 1
    plt.subplot(subplot)

    # 进行 5 轮迭代
    for i in range(5):
        # 训练 SVM，并传入样本权重
        svm_clf = SVC(kernel='rbf', C=0.05, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = svm_clf.predict(X_train)

        # 核心步骤：增加分错样本的权重
        # 权重更新公式：w = w * (1 + learning_rate)
        sample_weights[y_pred != y_train] *= (1 + learning_rate)

        # 绘制当前轮次的决策边界
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title(f'Manual AdaBoost (SVM)\nlearning_rate = {learning_rate}')

    # 绘制原始数据点
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")

plt.tight_layout()
plt.show()