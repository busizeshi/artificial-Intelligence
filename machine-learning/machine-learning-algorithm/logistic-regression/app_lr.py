import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logistic_regression

# 读取数据
data = pd.read_csv('./data/world-happiness-report-2017.csv')
columns=data.columns
print(columns)
to_remove = ['Happiness.Score', 'Happiness.Rank', 'Country']
features_name = [f for f in columns if f not in to_remove]
X = data[features_name]
# 将连续值目标变量转换为二进制分类标签
y = data['Happiness.Score']
threshold = y.median()  # 使用中位数作为阈值
y_binary = (y > threshold).astype(int)  # 高于中位数为1，否则为0
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)  # 使用二进制标签
print(X_train.shape)
print(X_test.shape)

model=logistic_regression.LogisticRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
print("准确率:",accuracy_score(y_test,y_predict))
model.plot_cost_history()