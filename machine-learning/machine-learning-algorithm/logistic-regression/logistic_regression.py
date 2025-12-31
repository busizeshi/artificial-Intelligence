import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        逻辑回归初始化
        :param learning_rate: 学习率
        :param num_iterations: 迭代次数
        """
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    @staticmethod
    def _sigmoid(x):
        """
        Sigmod函数
        :param x: 输入
        :return: 输出
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        训练模型
        :param X: 训练集
        :param y: 标签
        :return: 模型参数
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.num_iterations):
            # 前向转播，计算线性组合和激活值
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # 反向传播，计算梯度
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 记录损失
            if i % 100 == 0:
                cost = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
                self.cost_history.append(cost)

    def predict_proba(self, X):
        """
        预测标签
        :param X: 测试集
        :return: 预测标签
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        预测标签
        :param X: 测试集
        :param threshold: 阈值
        :return: 预测标签
        """
        return [1 if i > threshold else 0 for i in self.predict_proba(X)]

    def plot_cost_history(self):
        """
        绘制损失曲线
        :return: None
        """
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost History')
        plt.show()
