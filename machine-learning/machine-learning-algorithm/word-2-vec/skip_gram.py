import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 设置语料 (Demo用)
text = """
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data. 
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.
"""

# 简单的预处理
def preprocess(text):
    # 转小写，移除标点，分词
    text = text.lower().replace('.', '').replace(',', '').replace('\n', ' ')
    tokens = text.split()
    # 建立词表
    vocab = set(tokens)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return tokens, word2idx, idx2word, len(vocab)

tokens, word2idx, idx2word, vocab_size = preprocess(text)
print(f"词汇表大小: {vocab_size}")

# 生成 Skip-gram 训练对 (Center, Context)
def create_skipgram_dataset(tokens, window_size=2):
    data = []
    for i, target in enumerate(tokens):
        # 定义窗口范围
        context_indices = range(max(0, i - window_size), min(len(tokens), i + window_size + 1))
        for j in context_indices:
            if i != j:  # 跳过中心词本身
                data.append((word2idx[target], word2idx[tokens[j]]))
    return data

window_size = 2
training_data = create_skipgram_dataset(tokens, window_size)
print(f"前5个训练样本 (CenterIdx, ContextIdx): {training_data[:5]}")


# 2. 定义 Skip-gram 模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        # 中心词嵌入矩阵 (Input Embedding)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        # 上下文预测矩阵 (Output Linear Layer)
        self.linear = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, inputs):
        # inputs: [batch_size] -> [batch_size, embed_dim]
        embeds = self.embeddings(inputs)
        # output: [batch_size, vocab_size] (未归一化的分数/Logits)
        output = self.linear(embeds)
        return output


# 3. 训练配置
embed_dim = 10  # 词向量维度 (通常设为 100-300，演示用 10)
learning_rate = 0.01
epochs = 500

model = SkipGramModel(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()  # 包含 Softmax 和 Log计算
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 开始训练
print("开始训练...")
loss_history = []

for epoch in range(epochs):
    total_loss = 0

    # 将数据转换为 Tensor
    inputs = torch.tensor([x[0] for x in training_data], dtype=torch.long)
    labels = torch.tensor([x[1] for x in training_data], dtype=torch.long)

    # Forward
    optimizer.zero_grad()
    outputs = model(inputs)

    # Loss 计算
    loss = criterion(outputs, labels)

    # Backward
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    loss_history.append(total_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print("训练完成！")

# 提取训练好的词向量
# 注意：通常只取 Input Embeddings 作为最终词向量
word_vectors = model.embeddings.weight.data.numpy()


def get_similar_words(target_word, top_k=3):
    if target_word not in word2idx:
        return "Word not in vocabulary"

    target_idx = word2idx[target_word]
    target_vec = word_vectors[target_idx]

    similarities = []
    for i in range(vocab_size):
        if i == target_idx: continue

        vector = word_vectors[i]
        # 计算余弦相似度: (A . B) / (|A| * |B|)
        cosine_sim = np.dot(target_vec, vector) / (np.linalg.norm(target_vec) * np.linalg.norm(vector))
        similarities.append((idx2word[i], cosine_sim))

    # 排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# 测试评估
test_words = ["learning", "data", "science"]
print("\n--- 相似度评估 ---")
for w in test_words:
    print(f"'{w}' 的近义词: {get_similar_words(w)}")

from sklearn.manifold import TSNE


def plot_embeddings(word_vectors, idx2word):
    print("\n正在生成 t-SNE 可视化...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)  # perplexity需小于样本数
    vectors_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 12))
    for i, word in idx2word.items():
        x, y = vectors_2d[i]
        plt.scatter(x, y)
        plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.title("Skip-gram Word Embeddings Visualization")
    plt.grid(True)
    plt.show()


# 运行可视化
plot_embeddings(word_vectors, idx2word)