import numpy as np


def normalize(features):
    """
    该函数用于对输入的特征数据进行标准化处理（Z-score normalization），
    将数据转换为均值为0、标准差为1的分布。
    :param features: 输入的特征矩阵，通常为二维数组，形状为 (样本数, 特征数)
    :return: 标准化处理后的特征矩阵，以及原始特征的均值和标准差
    """
    features_normalized = np.copy(features).astype(np.float32)
    # 计算均值
    features_mean = np.mean(features, 0)
    # 计算标准差
    features_deviation = np.std(features, 0)
    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean
    # 防止除0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation


# 创建测试数据
def debug_normalize():
    # 示例1: 正常数据
    print("=== 示例1: 正常数据 ===")
    features1 = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
    print(f"原始数据:\n{features1}")

    # 添加断点或逐步执行
    normalized_features, mean, std = normalize(features1)

    print(f"均值: {mean}")
    print(f"标准差: {std}")
    print(f"标准化后数据:\n{normalized_features}")

    # 验证标准化结果
    print(f"标准化后均值: {np.mean(normalized_features, axis=0)}")
    print(f"标准化后标准差: {np.std(normalized_features, axis=0)}")

    print("\n=== 示例2: 包含常数特征 ===")
    features2 = np.array([[1, 5, 3],
                          [2, 5, 6],  # 第二列都是5
                          [3, 5, 9]])
    print(f"原始数据:\n{features2}")

    normalized_features2, mean2, std2 = normalize(features2)

    print(f"均值: {mean2}")
    print(f"标准差: {std2}")
    print(f"标准化后数据:\n{normalized_features2}")

    print("\n=== 示例3: 单个样本 ===")
    features3 = np.array([[1, 2, 3]])
    print(f"原始数据:\n{features3}")

    normalized_features3, mean3, std3 = normalize(features3)

    print(f"均值: {mean3}")
    print(f"标准差: {std3}")
    print(f"标准化后数据:\n{normalized_features3}")


# debug_normalize()