from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 1. 准备数据
X, y = datasets.load_wine(return_X_y=True) # 红酒数据集 (3类)
# 划分训练集和测试集 (这一步非常重要，不能把测试集拿去降维训练！)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 方案 A: 使用 PCA 的流水线 ===
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),        # 第一步：标准化 (必做)
    ('pca', PCA(n_components=2)),        # 第二步：PCA 降维
    ('clf', KNeighborsClassifier())      # 第三步：分类器
])

# === 方案 B: 使用 LDA 的流水线 ===
lda_pipeline = Pipeline([
    ('scaler', StandardScaler()),        # 第一步：标准化 (必做)
    ('lda', LinearDiscriminantAnalysis(n_components=2)), # 第二步：LDA 降维
    ('clf', KNeighborsClassifier())      # 第三步：分类器
])

# 3. 训练与评估
print("--- 效果对比 ---")

# 训练 PCA 方案
pca_pipeline.fit(X_train, y_train)
pca_score = pca_pipeline.score(X_test, y_test)
print(f"PCA + KNN 准确率: {pca_score:.4f}")

# 训练 LDA 方案
lda_pipeline.fit(X_train, y_train)
lda_score = lda_pipeline.score(X_test, y_test)
print(f"LDA + KNN 准确率: {lda_score:.4f}")

# 结论通常是：在有标签的分类任务中，LDA 往往比 PCA 效果好一点，或者用更少的维度达到同样的效果。