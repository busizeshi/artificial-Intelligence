from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

# 2. 定义第一层的基础模型（差异性越大越好）
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# 3. 定义第二层的元模型（通常使用简单的模型，如逻辑回归）
meta_model = LogisticRegression()

# 4. 构建 Stacking 模型
stack_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # 内部自动进行 5 折交叉验证
)

# 5. 训练与评估
stack_clf.fit(X_train, y_train)
print(f"Stacking 模型测试集准确率: {stack_clf.score(X_test, y_test):.4f}")
