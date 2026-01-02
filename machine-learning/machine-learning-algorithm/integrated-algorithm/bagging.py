import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# 加载数据
iris = datasets.load_iris()
# 为了方便可视化，只取前两个特征：花萼长度(Sepal length)和花萼宽度(Sepal width)
X = iris.data[:, :2]
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. 单个决策树（基模型）
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# 2. Bagging
# 使用 50 棵决策树
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=50,
    max_samples=1.0, bootstrap=True, random_state=42
)
bag_clf.fit(X_train, y_train)

# 3. 随机森林
# n_estimators 表示森林中树的数量
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
rf_clf.fit(X_train, y_train)

models = {"Decision Tree": dt_clf, "Bagging": bag_clf, "Random Forest": rf_clf}

for name, clf in models.items():
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")


def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.title(title)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')


# 绘图
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plot_decision_boundary(dt_clf, X, y, "Single Decision Tree")

plt.subplot(1, 3, 2)
plot_decision_boundary(bag_clf, X, y, "Bagging (50 Trees)")

plt.subplot(1, 3, 3)
plot_decision_boundary(rf_clf, X, y, "Random Forest (50 Trees)")

plt.tight_layout()
plt.show()