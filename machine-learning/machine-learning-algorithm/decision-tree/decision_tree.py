import numpy as np
from collections import Counter


class Node:
    """决策树节点类"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # 分裂的特征索引
        self.threshold = threshold  # 分裂的阈值（本例主要处理类别，可理解为特征值）
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 如果是叶子节点，存储类别标签

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """训练入口"""
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # 1. 停止条件：标签纯净、达到最大深度或样本数不足
        if (n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 2. 贪心搜索最佳分裂特征
        feat_idxs = np.arange(n_feats)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # 3. 创建子节点
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # 处理无法进一步分裂的情况
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # 计算信息增益
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """计算信息增益: Gain = Entropy(parent) - [weighted_avg * Entropy(children)]"""
        # 父节点熵
        parent_entropy = self._entropy(y)

        # 创建子集
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # 计算子节点的加权平均熵
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # 信息增益
        return parent_entropy - child_entropy

    def _split(self, X_column, split_threshold):
        """根据阈值划分为左右两部分"""
        # 这里为了演示方便，处理类别数据：等于 threshold 为左，不等于为右
        left_idxs = np.where(X_column == split_threshold)[0]
        right_idxs = np.where(X_column != split_threshold)[0]
        return left_idxs, right_idxs

    def _entropy(self, y):
        """计算熵: H(S) = -sum(p * log2(p))"""
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """返回出现次数最多的标签"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """预测入口"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] == node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# --- 测试代码 (模拟 Play Tennis 数据集) ---
if __name__ == "__main__":
    # 特征映射转换 (由于 numpy 偏好数字，我们将类别映射为 ID)
    # Outlook: Sunny=0, Overcast=1, Rain=2
    # Humidity: High=0, Normal=1
    # Wind: Weak=0, Strong=1
    # Play: No=0, Yes=1

    X = np.array([
        [0, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0], [2, 1, 0],
        [2, 1, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0], [2, 1, 0],
        [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 0, 1]
    ])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

    # 训练
    clf = DecisionTree(max_depth=10)
    clf.fit(X, y)

    # 预测一个 Sunny, Normal humidity, Strong wind 的天气 (0, 1, 1)
    test_sample = np.array([[0, 1, 1]])
    prediction = clf.predict(test_sample)

    label_map = {0: "No (不打球)", 1: "Yes (去打球)"}
    print(f"预测结果: {label_map[prediction[0]]}")
