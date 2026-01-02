"""
贝叶斯算法
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# 1. 加载数据
iris = datasets.load_iris()
X = iris.data  # 特征 (花萼长度等)
y = iris.target # 类别 (三种花)

# 2. 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 初始化模型 (这里选择高斯朴素贝叶斯，因为特征是连续数值)
model = GaussianNB()

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 简单的结果打印
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
