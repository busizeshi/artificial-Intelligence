import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. 加载完整数据
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. 训练随机森林
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X, y)

# 3. 获取特征重要性
importances = rf_clf.feature_importances_
feature_names = iris.feature_names

# 4. 整理成 DataFrame 方便绘图
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# 5. 可视化
plt.figure(figsize=(10, 6))
feature_imp.plot(kind='barh', color='skyblue')
plt.title('Feature Importances in Random Forest (Iris Dataset)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # 最高的重要性排在最上面
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
