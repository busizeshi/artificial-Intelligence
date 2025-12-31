import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        模型初始化
        :param k: 簇个数
        :param max_iters: 迭代次数
        :param tol: 容差参数
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.tol = tol

    def fit(self, X):
        """
        训练模型
        :param X: 训练集
        :return: 模型参数
        """
        # 随机初始化聚类中心
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. 分配步骤：计算每个点到所有中心的欧氏距离
            # 距离公式: d = sqrt(sum((x - c)^2))
            distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distance, axis=1)

            # 3. 更新步骤：计算每个簇的新中心
            new_centroids = np.array([
                X[self.labels == j].mean(axis=0) if len(X[self.labels == j]) > 0
                else self.centroids[j]  # 防止某个簇为空
                for j in range(self.k)
            ])

            # 4. 检测收敛 (中心变化小于阈值)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"迭代次数: {i + 1} 处收敛")
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        预测标签
        :param X: 测试集
        :return: 预测标签
        """
        distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distance, axis=1)


if __name__ == "__main__":
    # 生成模拟数据
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # 训练模型
    clf = KMeans(k=4)
    clf.fit(X)

    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=clf.labels, s=30, cmap='viridis')
    plt.scatter(clf.centroids[:, 0], clf.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.title("K-Means Implementation from Scratch")
    plt.legend()
    plt.show()
