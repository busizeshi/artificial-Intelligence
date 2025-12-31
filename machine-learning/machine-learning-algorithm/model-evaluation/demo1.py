import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# 1. 加载数据 (二分类: 恶性/良性)
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 训练模型并获取预测概率
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
# 获取预测为正类的概率 (第1列)
y_probs = model.predict_proba(X_test)[:, 1]

# 3. 获取 ROC 曲线的三个关键数组
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# 4. 挑出几个特定的阈值观察其在曲线上的位置
sample_thresholds = [0.1, 0.5, 0.9]
points = []
for t in sample_thresholds:
    # 找到最接近指定阈值的索引
    idx = np.argmin(np.abs(thresholds - t))
    points.append((fpr[idx], tpr[idx], thresholds[idx]))

# 5. 绘图
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc(fpr, tpr):.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # 对角线

# 标注特定阈值点
for f, t, th in points:
    plt.scatter(f, t, color='red', s=100, zorder=5)
    plt.text(f+0.02, t-0.05, f"Threshold: {th:.2f}\n(FPR:{f:.2f}, TPR:{t:.2f})", fontsize=9)

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve with Threshold Samples')
plt.legend()
plt.grid(alpha=0.3)
plt.show()