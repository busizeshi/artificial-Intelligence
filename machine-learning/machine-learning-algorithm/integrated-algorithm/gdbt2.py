import numpy as np
import joblib  # 用于模型导出与加载
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 准备模拟数据
X = np.random.rand(200, 1) - 0.5
y = 4 * X[:, 0]**2 + 0.1 * np.random.randn(200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 寻找最佳树的数量 (Early Stopping)
# 训练一个足够大的模型（例如 200 棵树）
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt.fit(X_train, y_train)

# 使用 staged_predict 找到验证集误差最小的点
errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
best_n_estimators = np.argmin(errors) + 1
print(f"最佳树的数量: {best_n_estimators}")

# 3. 使用最优参数重新训练模型
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators, learning_rate=0.1, random_state=42)
gbrt_best.fit(X_train, y_train)

# 4. 导出模型 (Save)
model_filename = "gbdt_best_model.pkl"
joblib.dump(gbrt_best, model_filename)
print(f"模型已导出为: {model_filename}")

# --- 模拟另一段程序或部署环境 ---

# 5. 加载模型 (Load)
loaded_model = joblib.load(model_filename)
print("模型加载成功！")

# 6. 使用加载的模型进行预测
new_data = np.array([[0.3], [-0.2], [0.1]])
predictions = loaded_model.predict(new_data)
print(f"对新数据的预测结果: {predictions}")

"""
提前停止
"""
# 配置自动提前停止
# 利用 n_iter_no_change（最推荐，最自动化）
gbrt = GradientBoostingRegressor(
    n_estimators=1000,      # 设置一个很大的上限
    learning_rate=0.1,
    n_iter_no_change=10,    # 如果连续 10 轮验证分数没有改善，则停止
    validation_fraction=0.1,# 拿 10% 的数据作为内部验证集
    tol=1e-4,               # 改善幅度小于这个值视为没有改善
    random_state=42
)

gbrt.fit(X_train, y_train)

print(f"训练在第 {len(gbrt.estimators_)} 棵树时停止了。")