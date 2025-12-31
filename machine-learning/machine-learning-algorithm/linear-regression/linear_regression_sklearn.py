import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. 获取数据集
# 建议加上 error_bad_lines=False 以防万一，或者确保路径正确
data = pd.read_csv('./data/world-happiness-report-2017.csv')

# 2. 特征选择优化
# 自动选择数值列
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

# 【关键点】移除与目标强相关的非解释性特征，避免数据泄露
# Whisker.high/low 是得分的置信区间，不能作为预测特征
to_remove = ['Happiness.Score', 'Happiness.Rank', 'Whisker.high', 'Whisker.low']
features_name = [f for f in numeric_features if f not in to_remove]

print(f"使用的特征数量: {len(features_name)}")
print("使用的特征列:", features_name)

# 3. 分割数据集
X = data[features_name]
y = data['Happiness.Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 构造 Pipeline (标准化 + SGD)
# 使用 Pipeline 可以确保测试集使用训练集的缩放标准，防止数据泄露
model_pipeline = Pipeline([
    ('scaler', StandardScaler()), # 自动进行数据标准化
    ('sgd_reg', SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

# 5. 训练模型
model_pipeline.fit(X_train, y_train)

# 6. 预测与评估
y_predict = model_pipeline.predict(X_test)

r2_score_val = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print("-" * 30)
print(f"模型评估 R2 Score: {r2_score_val:.4f}")
print(f"均方误差 MSE: {mse:.4f}")

# 7. 查看回归方程系数 (从 Pipeline 中提取)
# 注意：因为多了 scaler 层，我们需要先访问 'sgd_reg' 步骤
final_model = model_pipeline.named_steps['sgd_reg']
print("-" * 30)
print("权重系数 (Weights):")
for name, coef in zip(features_name, final_model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"截距 (Intercept): {final_model.intercept_[0]:.4f}")

# 8. 导出完整流水线
# 这样你在加载模型后，不需要手动对新输入的数据做标准化
joblib.dump(model_pipeline, './model/sklearn_sgd_pipeline.pkl')

print("-" * 30)
print("模型流水线已成功保存。")

# --- 验证加载 ---
loaded_pipe = joblib.load('./model/sklearn_sgd_pipeline.pkl')
# 直接喂入原始测试集（无需手动缩放）
test_pred = loaded_pipe.predict(X_test)
print(f"验证加载后的 R2: {r2_score(y_test, test_pred):.4f}")