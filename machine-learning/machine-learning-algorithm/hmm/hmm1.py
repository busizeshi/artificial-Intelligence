import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# ==========================================
# 1. 准备数据：制造一个虚假的“股票市场”
# ==========================================
np.random.seed(42)

# 假设我们有 1000 天的数据
n_samples = 1000

# 定义两个真实的隐状态：
# 状态 0 (平稳期): 均值 0，方差小 (0.5)
# 状态 1 (动荡期): 均值 0，方差大 (2.0)
# 我们先自己生成数据，这样后面可以验证模型准不准

# 真实的转换矩阵 (Stay probability high)
# 0 -> 0: 95%, 0 -> 1: 5%
# 1 -> 1: 90%, 1 -> 0: 10%
true_transmat = np.array([[0.95, 0.05],
                          [0.10, 0.90]])

# 生成模拟的隐状态序列 (Z) 和 观测序列 (X)
# 这里手动模拟马尔可夫过程
true_hidden_states = [0]
observations = []

for i in range(1, n_samples):
    # 根据上一个状态决定当前状态
    prev_state = true_hidden_states[-1]
    current_state = np.random.choice([0, 1], p=true_transmat[prev_state])
    true_hidden_states.append(current_state)

    # 根据当前状态生成观测值 (收益率)
    if current_state == 0:
        obs = np.random.normal(0, 0.5) # 平稳
    else:
        obs = np.random.normal(0, 2.0) # 动荡
    observations.append([obs])

observations = np.array(observations)
true_hidden_states = np.array(true_hidden_states)

print(f"数据生成完毕。前10天观测值: {observations[:5].flatten()}")

# ==========================================
# 2. 训练 HMM 模型
# ==========================================
# 我们使用 GaussianHMM，因为观测值(收益率)是连续数值，符合高斯分布
# n_components=2 表示我们假设有两个隐状态
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)

# 训练 (Fit) - 这一步使用 Baum-Welch 算法
model.fit(observations)

# ==========================================
# 3. 预测 (解码)
# ==========================================
# 也就是使用 Viterbi 算法，找出最可能的隐状态序列
predicted_states = model.predict(observations)

# ==========================================
# 4. 评估与可视化
# ==========================================
# 检查模型是否学会了正确的参数
print("\n--- 模型参数评估 ---")
print(f"学习到的转移矩阵:\n{model.transmat_}")
print(f"真实的转移矩阵:\n{true_transmat}")

print(f"\n学习到的各状态均值:\n{model.means_}")
print(f"学习到的各状态方差:\n{model.covars_}")

# 纠正标签翻转问题 (Label Switching)
# HMM 可能会把状态 0 标记为 1，状态 1 标记为 0，这是无监督学习的特性。
# 我们根据方差大小来对齐标签：方差大的定义为"动荡期(1)"
if model.covars_[0][0] > model.covars_[1][0]:
    # 如果状态0的方差比状态1大，说明模型搞反了，我们要翻转一下
    predicted_states = 1 - predicted_states
    print("\n(检测到标签翻转，已自动修正)")

# 确保数组长度一致
min_length = min(len(predicted_states), len(true_hidden_states))
predicted_states = predicted_states[:min_length]
true_hidden_states = true_hidden_states[:min_length]

# 计算准确率
accuracy = np.mean(predicted_states == true_hidden_states)
print(f"\nHMM 状态识别准确率: {accuracy:.2%}")

# 画图
plt.figure(figsize=(15, 6))

# 画出观测值 (股价波动)
plt.plot(observations, label='Daily Returns', alpha=0.6, color='grey')

# 用背景色表示预测的隐状态
# 找出所有被预测为状态 1 (动荡期) 的区域
y_lim = plt.ylim()
for i, state in enumerate(predicted_states):
    if state == 1:
        plt.fill_between([i, i+1], y_lim[0], y_lim[1], color='red', alpha=0.2)

# 添加图例说明
plt.plot([], [], color='red', alpha=0.2, linewidth=10, label='Predicted High Volatility (Hidden State)')
plt.title(f"HMM Regime Detection (Accuracy: {accuracy:.2%})\nRed background indicates 'High Volatility' State detected by HMM", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()