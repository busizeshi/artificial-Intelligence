import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. 获取数据集
data = pd.read_csv('./data/world-happiness-report-2017.csv')

# 2. 数据处理：将连续变量转换为分类变量（二分类）
# 我们取中位数作为阈值：高于中位数为 1 (高幸福感)，低于为 0 (低幸福感)
threshold = data['Happiness.Score'].median()
data['Happiness.Level'] = (data['Happiness.Score'] > threshold).astype(int)

# 3. 选择数值型列作为特征
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
# 移除目标变量和原始得分相关的列
to_remove = ['Happiness.Score', 'Happiness.Rank', 'Happiness.Level']
features_name = [f for f in numeric_features if f not in to_remove]

print(f"使用的特征列: {features_name}")
print(f"分类阈值 (中位数): {threshold}")

# 4. 分割数据集
X = data[features_name]
y = data['Happiness.Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 初始化并训练逻辑回归模型
# 使用 solver='liblinear' 适合这种中小型数据集
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# 6. 预测
y_predict = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # 获取预测为“高幸福感”的概率

# 7. 评估模型
print("-" * 30)
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_predict):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_predict))

# 8. 查看特征重要性（系数）
# 逻辑回归的系数反映了每个特征对分类结果的贡献程度
importance = pd.DataFrame({'Feature': features_name, 'Coef': model.coef_[0]})
print("\n特征系数（绝对值越大越重要）:")
print(importance.sort_values(by='Coef', ascending=False))

# 9. 可视化 ROC 曲线
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve for Happiness Classification")
plt.show()