import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# 1. 生成数据
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, s=50, cmap='viridis')
plt.show()
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=0)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, s=50, cmap='viridis')
plt.show()
X_moons = StandardScaler().fit_transform(X_moons) # 数据标准化


# 2. 定义函数
def find_optimal_k(X):
    inertias = []
    silhouettes = []
    k_range = range(2, 11)

    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig_k, (ax1_k, ax2_k) = plt.subplots(1, 2, figsize=(12, 4))

    ax1_k.plot(k_range, inertias, 'bo-')
    ax1_k.set_title('Elbow Method (Inertia)')
    ax1_k.set_xlabel('Number of clusters k')
    ax1_k.set_ylabel('Inertia')

    ax2_k.plot(k_range, silhouettes, 'ro-')
    ax2_k.set_title('Silhouette Scores')
    ax2_k.set_xlabel('Number of clusters k')
    ax2_k.set_ylabel('Score')
    plt.tight_layout()
    plt.show()


# 3. 使用函数
find_optimal_k(X_blobs)

# 4. 使用DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)
y_km = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10).fit_predict(X_moons)
y_db = db.fit_predict(X_moons)

# 4. 可视化结果
fig_viz, (ax1_viz, ax2_viz) = plt.subplots(1, 2, figsize=(12, 5))

ax1_viz.scatter(X_moons[:, 0], X_moons[:, 1], c=y_km, cmap='viridis', s=30)
ax1_viz.set_title(f'K-Means (ARI: {adjusted_rand_score(y_moons, y_km):.2f})')

ax2_viz.scatter(X_moons[:, 0], X_moons[:, 1], c=y_db, cmap='plasma', s=30)
ax2_viz.set_title(f'DBSCAN (ARI: {adjusted_rand_score(y_moons, y_db):.2f})')

plt.show()
