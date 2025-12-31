import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap

# 设置绘图字体和风格
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ==========================================
# 第一部分：分类任务 (Play Tennis)
# ==========================================
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
encoders = {}
df_encoded = pd.DataFrame()
for column in df.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    encoders[column] = le

X_play = df_encoded.drop('Play', axis=1)
y_play = df_encoded['Play']
clf_play = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_play.fit(X_play, y_play)

# ==========================================
# 第二部分：决策边界可视化 (Iris Dataset)
# ==========================================
iris = load_iris()
X_iris = iris.data[:, 2:]  # 仅取花瓣长度和宽度
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)


def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris_flag=True, legend=False):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    # 背景着色
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

    # 绘制训练点
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris-Virginica")
    plt.axis(axes)

    if iris_flag:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    if legend:
        plt.legend(loc="lower right", fontsize=14)


plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X_iris, y_iris)
# 手动标出主要分割线以对应 Depth 概念
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.title('Decision Tree Decision Boundaries (Iris)')
plt.show()

# ==========================================
# 第三部分：回归任务 (Regression)
# ==========================================
# 生成带噪声的正弦波数据
np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + 0.1 * np.random.randn(80)

# 训练两个不同深度的回归树
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X_reg, y_reg)
tree_reg2.fit(X_reg, y_reg)


def plot_regression_predictions(tree_reg, X, y, axes=[0, 5, -1.2, 1.2], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")


plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_regression_predictions(tree_reg1, X_reg, y_reg)
plt.title("max_depth=2 (Underfitting)", fontsize=14)
plt.legend(loc="upper center", fontsize=18)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X_reg, y_reg, ylabel=None)
plt.title("max_depth=3 (Better Fit)", fontsize=14)

plt.tight_layout()
plt.show()

# 回归模型评价指标
y_reg_pred = tree_reg2.predict(X_reg)
print(f"--- 回归任务评估 (Depth=3) ---")
print(f"均方误差 (MSE): {mean_squared_error(y_reg, y_reg_pred):.4f}")
print(f"R2 分数: {r2_score(y_reg, y_reg_pred):.4f}")