import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. 加载数据 (鸢尾花数据集)
iris = datasets.load_iris()
X = iris.data  # 4个特征：花萼长、宽，花瓣长、宽
y = iris.target # 3个类别：0, 1, 2
target_names = iris.target_names

# 2. 运行 PCA (作为对比)
# PCA 是无监督的，它不知道 y (类别) 的存在
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. 运行 LDA (主角)
# LDA 是有监督的，它必须利用 y 来计算如何把类别分开
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# 4. 可视化对比
plt.figure(figsize=(14, 6))

# 定义颜色和标记
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

# === 画左图：PCA 结果 ===
plt.subplot(1, 2, 1)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, lw=lw,
                label=target_name)
plt.title('PCA of IRIS dataset\n(Maximizes Variance)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, linestyle=':', alpha=0.6)

# === 画右图：LDA 结果 ===
plt.subplot(1, 2, 2)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=0.8, lw=lw,
                label=target_name)
plt.title('LDA of IRIS dataset\n(Maximizes Class Separation)')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# 5. 输出方差解释率 (解释 LDA 保留了多少分类信息)
print("LDA 投影后的解释方差比 (Explained Variance Ratio):")
print(lda.explained_variance_ratio_)