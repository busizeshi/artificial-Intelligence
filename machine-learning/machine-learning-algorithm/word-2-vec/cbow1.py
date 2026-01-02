import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
plt.rcParams['font.sans-serif'] = ['SimHei']

# ==========================================
# 第一步：准备“假”数据 (模拟训练好的 Word2Vec 模型)
# ==========================================
# 在真实场景中，这些 vectors 来自 model.wv[word]
def create_mock_model():
    np.random.seed(42)
    dim = 50  # 假设词向量是 50 维

    # 我们定义三个“圈子”，每个圈子里的词向量应该靠得很近
    clusters = {
        '动画片': ['玩具总动员', '皮克斯', '迪士尼', '卡通', '米老鼠'],
        '动作片': ['勇敢者游戏', '冒险', '战斗', '英雄', '爆炸'],
        '爱情片': ['罗密欧', '恋爱', '接吻', '婚礼', '玫瑰']
    }

    words = []
    vectors = []
    labels = []  # 用于画图上色

    for label, word_list in clusters.items():
        # 每个类别随机生成一个“中心向量”
        center_vec = np.random.randn(dim)
        for word in word_list:
            words.append(word)
            labels.append(label)
            # 词向量 = 中心 + 随机噪声 (模拟语义接近)
            vec = center_vec + np.random.normal(0, 0.2, dim)
            vectors.append(vec)

    return words, np.array(vectors), labels


words, vectors, labels = create_mock_model()
print(f"数据准备完毕：共 {len(words)} 个词，向量维度 {vectors.shape[1]}")

# ==========================================
# 第二步：核心分析功能 —— t-SNE 降维可视化
# ==========================================
# 业界标准操作：将高维向量降维到 2D 以便肉眼观察
print("正在进行 t-SNE 降维...")
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectors_2d = tsne.fit_transform(vectors)

# 开始画图
plt.figure(figsize=(10, 8))
# 定义颜色映射
color_map = {'动画片': 'red', '动作片': 'blue', '爱情片': 'green'}

# 绘制散点
for i, word in enumerate(words):
    category = labels[i]
    x, y = vectors_2d[i]
    plt.scatter(x, y, c=color_map[category], s=100, alpha=0.6)
    # 给点加上文字标签，防止重叠微调一下位置
    plt.text(x + 0.2, y + 0.2, word, fontsize=12, fontproperties='SimHei')  # 注意：Mac/Linux可能需要设置中文字体

plt.title("Word2Vec 词向量空间可视化 (t-SNE)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# ==========================================
# 第三步：业务验证 —— 寻找 "最相似" (Top-N)
# ==========================================
# 比如用户搜了 "玩具总动员"，系统该推什么？
target = '玩具总动员'
target_idx = words.index(target)
target_vec = vectors[target_idx].reshape(1, -1)

# 计算余弦相似度
scores = cosine_similarity(target_vec, vectors)[0]

# 排序 (argsort 返回的是索引，[::-1] 表示倒序)
sorted_indices = np.argsort(scores)[::-1]

print(f"\n--- 搜索词：'{target}' ---")
print("推荐结果 (相似度 Top 5):")
for idx in sorted_indices[1:6]:  # [1:6] 是因为第0个肯定是它自己(相似度1.0)
    print(f"{words[idx]}: {scores[idx]:.4f}")