import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. 准备数据
wine = datasets.load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"原始数据形状: {X.shape} (13个特征)")

# 划分训练集 (70%) 和测试集 (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 方案 A: 全特征直接训练 (不降维)
# ==========================================
start_time = time.time()

pipe_full = Pipeline([
    ('scaler', StandardScaler()),       # 必须标准化
    ('clf', LogisticRegression())       # 分类器
])

pipe_full.fit(X_train, y_train)
y_pred_full = pipe_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)
time_full = time.time() - start_time

# ==========================================
# 方案 B: PCA 降维后训练 (只保留 2 个主成分)
# ==========================================
start_time = time.time()

pipe_pca = Pipeline([
    ('scaler', StandardScaler()),       # 必须标准化
    ('pca', PCA(n_components=2)),       # 降维：13 -> 2
    ('clf', LogisticRegression())       # 分类器
])

pipe_pca.fit(X_train, y_train)
y_pred_pca = pipe_pca.predict(X_test)
acc_pca = accuracy_score(y_test, y_pred_pca)
time_pca = time.time() - start_time

# ==========================================
# 3. 评估与对比
# ==========================================
print("\n" + "="*40)
print("       模型评估结果对比       ")
print("="*40)

print(f"【方案 A: 使用全部 13 个特征】")
print(f"准确率: {acc_full:.2%}")
print(f"耗时:   {time_full:.4f} 秒")
print("-" * 20)

print(f"【方案 B: 使用 PCA (2维) 特征】")
print(f"准确率: {acc_pca:.2%}")
print(f"耗时:   {time_pca:.4f} 秒")
print("-" * 20)

# 看看 PCA 到底保留了多少原始信息？
pca_step = pipe_pca.named_steps['pca']
explained_variance = np.sum(pca_step.explained_variance_ratio_)
print(f"注意：PCA 仅用 2 个特征就保留了原始数据 {explained_variance:.2%} 的信息量。")

# ==========================================
# 4. 可视化：为什么 2 个特征也能分得这么好？
# ==========================================
# 因为我们降到了 2 维，所以可以直接画出来！
X_test_transformed = pipe_pca[:-1].transform(X_test) # 只做前两步处理(缩放+PCA)

plt.figure(figsize=(10, 6))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_test_transformed[y_test == i, 0], 
                X_test_transformed[y_test == i, 1], 
                color=color, alpha=0.8, lw=2,
                label=target_name)

plt.title('PCA Result visualization (Test Set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# 打印详细分类报告 (PCA版本)
print("\nPCA 模型的详细分类报告:\n")
print(classification_report(y_test, y_pred_pca, target_names=target_names))