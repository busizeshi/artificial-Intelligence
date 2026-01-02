from gensim.models import Word2Vec

# 1. 准备语料
sentences = [
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['The', 'lazy', 'dog', 'sleeps', 'on', 'the', 'mat'],
    ['Foxes', 'are', 'quick', 'and', 'brown']
]

# 2. 训练 CBOW 模型
# sg=0 : 明确指定使用 CBOW
# window=2 : 也就是看左边2个词，右边2个词
# vector_size=100：词向量的维度，设置为100维向量表示每个词语
model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=0)

# 3. 验证
# 虽然数据量很小，但模型应该能学到 brown 和 quick/fox 的关系
vector = model.wv['brown']
print("词向量维度:", len(vector))

# 4. 核心应用：根据上下文猜词 (predict_output_word 是 gensim 的一个功能)
# 给定上下文：['quick', 'fox']，猜中间可能是啥
# 注意：这个函数在较新版本的 gensim 中可用，用于模拟 CBOW 的预测过程
# topn：表示返回最可能的前n个结果
try:
    print("\n上下文 ['quick', 'fox'] 中间可能的词:")
    print(model.predict_output_word(['quick', 'fox'], topn=3))
except AttributeError:
    print("你的 Gensim 版本可能不支持直接调用 predict_output_word，但原理是一样的。")
# [('dog', np.float32(0.06666704)), ('brown', np.float32(0.06666689)), ('lazy', np.float32(0.06666685))]