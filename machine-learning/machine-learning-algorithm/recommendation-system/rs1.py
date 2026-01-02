import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. 模拟数据：5个用户对5个物品的打分 (0表示没看过)
data = {
    'User_A': [5, 3, 0, 1, 0], # 喜欢 钢铁侠、泰坦尼克
    'User_B': [4, 0, 0, 1, 0], # 喜欢 钢铁侠
    'User_C': [1, 1, 0, 5, 0], # 喜欢 哆啦A梦
    'User_D': [0, 0, 5, 4, 4], # 喜欢 哆啦A梦、蜡笔小新
    'User_E': [0, 0, 5, 0, 4], # 喜欢 蜡笔小新
}
# 物品列表: ['钢铁侠', '泰坦尼克', '蜡笔小新', '哆啦A梦', '死侍']
df = pd.DataFrame(data, index=['Item_1', 'Item_2', 'Item_3', 'Item_4', 'Item_5'])

print("原始评分矩阵 (行=物品, 列=用户):")
print(df)

# 2. 计算物品相似度 (Item-Item Similarity)
# 我们看哪两个物品的行向量长得像
item_sim_matrix = cosine_similarity(df)
item_sim_df = pd.DataFrame(item_sim_matrix, index=df.index, columns=df.index)

print("\n物品相似度矩阵 (Item 1 和 Item 2 很像，因为都被User A高分):")
print(item_sim_df.round(2))

# 3. 简单的推荐逻辑
# 假设 我 (User_New) 喜欢 'Item_1' (钢铁侠)，系统该推荐什么？
liked_item = 'Item_1'

# 找到与 'Item_1' 最相似的物品（排除自己）
recommendations = item_sim_df[liked_item].drop(liked_item).sort_values(ascending=False)

print(f"\n因为你喜欢 {liked_item}，推荐给你:")
print(recommendations.head(2))