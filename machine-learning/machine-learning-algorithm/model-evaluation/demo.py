"""
模型评估方法
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score

np.random.seed(42)

# 数据集读取
mnist = fetch_openml('mnist_784', version=1, as_frame=False)  # 使用 fetch_openml 替代 fetch_mldata
print("数据集大小:", mnist.data.shape)
# plt.imshow(mnist.data[0].reshape(28,28), cmap="binary")
# plt.show()

X,y=mnist.data,mnist.target
shuffle_index=np.random.permutation(len(X))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("训练集大小:", X_train.shape)

"""
交叉验证
"""
model=SGDClassifier(random_state=42, max_iter=1000)  # 增加最大迭代次数
model.fit(X_train,y_train)
# print("训练集准确率:", model.score(X_train,y_train))
# print("测试集准确率:", model.score(X_test,y_test))
# scores=cross_val_score(model,X_train,y_train,cv=3)
# print("交叉验证准确率:", scores)

"""
混淆矩阵
默认阈值=0.5
"""
y_pred=model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
plt.matshow(confusion_matrix(y_test,y_pred))
plt.show()
# 精确率衡量的是模型预测为正类的样本中，实际为正类的比例
print(precision_score(y_test,y_pred,average="macro"))
# 召回率衡量的是实际正类样本中，被模型正确预测为正类的比例
print(recall_score(y_test,y_pred,average="macro"))
# F1 分数是精确率和召回率的调和平均值
print(f1_score(y_test,y_pred,average="macro"))