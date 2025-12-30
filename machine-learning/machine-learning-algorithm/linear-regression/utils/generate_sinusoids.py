import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    该函数用于生成正弦特征，将原始数据通过不同频率的正弦函数进行扩展，增加特征维度。
    :param dataset: 输入的数据集，通常为二维数组
    :param sinusoid_degree: 正弦函数的最高次数，决定生成多少种不同频率的正弦特征
    :return: 返回所有生成的正弦特征组成的矩阵
    """
    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids

def debug_generate_sin():
    print("=== 调试 generate_sin 函数 ===")

    # 示例1: 简单一维数据
    print("\n1. 一维数据示例:")
    dataset1 = np.array([[0], [np.pi / 2], [np.pi], [3 * np.pi / 2], [2 * np.pi]])
    print(f"原始数据:\n{dataset1}")
    print(f"数据形状: {dataset1.shape}")

    # 调试参数
    sinusoid_degree = 3
    print(f"正弦度数: {sinusoid_degree}")

    # 逐步执行函数
    num_examples = dataset1.shape[0]
    print(f"样本数量: {num_examples}")

    # 初始化空数组
    sinusoids = np.empty((num_examples, 0))
    print(f"初始空数组形状: {sinusoids.shape}")

    # 循环生成不同频率的正弦特征
    for degree in range(1, sinusoid_degree + 1):
        print(f"\n处理度数 {degree}:")
        sinusoid_features = np.sin(degree * dataset1)
        print(f"  sin({degree} * dataset) = \n{sinusoid_features}")
        print(f"  当前sinusoids形状: {sinusoids.shape}")

        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
        print(f"  拼接后sinusoids形状: {sinusoids.shape}")

    print(f"\n最终结果:\n{sinusoids}")

    # 验证结果
    result = generate_sin(dataset1, sinusoid_degree)
    print(f"函数返回结果:\n{result}")
    print(f"结果匹配: {np.allclose(result, sinusoids)}")

    # 示例2: 二维数据
    print("\n\n2. 二维数据示例:")
    dataset2 = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"原始数据:\n{dataset2}")

    result2 = generate_sin(dataset2, 2)
    print(f"正弦度数为2的结果:\n{result2}")
    print(f"结果形状: {result2.shape}")

    # 逐步分析
    print("\n逐步分析:")
    print(f"度数1: sin(1*dataset) = sin(dataset) = \n{np.sin(dataset2)}")
    print(f"度数2: sin(2*dataset) = \n{np.sin(2 * dataset2)}")
    print(f"拼接后: 度数1 | 度数2")

# debug_generate_sin()
