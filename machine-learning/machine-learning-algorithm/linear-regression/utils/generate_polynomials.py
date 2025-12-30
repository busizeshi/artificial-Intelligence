"""Add polynomial features to the features set"""

import numpy as np
from . import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """变换方法：
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    """

    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    polynomials = np.empty((num_examples_1, 0))

    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize.normalize(polynomials)[0]

    return polynomials


def debug_generate_polynomials():
    print("=== 调试 generate_polynomials 函数 ===")

    # 示例数据
    dataset = np.array([[1, 2],
                        [3, 4],
                        [5, 6]], dtype=float)
    polynomial_degree = 2

    print(f"输入数据:\n{dataset}")
    print(f"数据形状: {dataset.shape}")
    print(f"多项式度数: {polynomial_degree}")

    # 步骤1: 分割数据
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    print(f"\n步骤1 - 数据分割:")
    print(f"dataset_1:\n{dataset_1}")
    print(f"dataset_2:\n{dataset_2}")

    # 步骤2: 获取形状
    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    print(f"\n步骤2 - 数据形状:")
    print(f"dataset_1 形状: ({num_examples_1}, {num_features_1})")
    print(f"dataset_2 形状: ({num_examples_2}, {num_features_2})")

    # 步骤3: 特征数均衡处理
    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    print(f"\n步骤3 - 特征数均衡后:")
    print(f"dataset_1:\n{dataset_1}")
    print(f"dataset_2:\n{dataset_2}")

    # 步骤4: 初始化结果数组
    polynomials = np.empty((num_examples_1, 0))
    print(f"\n步骤4 - 初始化结果数组形状: {polynomials.shape}")

    # 步骤5: 生成多项式特征
    print(f"\n步骤5 - 生成多项式特征:")
    for i in range(1, polynomial_degree + 1):
        print(f"  处理 {i} 次项:")
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            print(f"    j={j}: ({i - j}次项 * {j}次项) = {i}次项")
            print(f"      dataset_1^{i - j} * dataset_2^{j} = \n{polynomial_feature}")

            original_shape = polynomials.shape
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)
            print(f"      拼接前形状: {original_shape}, 拼接后形状: {polynomials.shape}")

    print(f"\n最终多项式特征:\n{polynomials}")
    print(f"最终形状: {polynomials.shape}")

    # 完整函数调用
    result = generate_polynomials(dataset, polynomial_degree)
    print(f"\n函数返回结果:\n{result}")
    print(f"结果形状: {result.shape}")

    # 验证结果是否一致
    print(f"结果匹配: {np.array_equal(polynomials, result)}")


def debug_with_different_degrees():
    print("\n=== 不同度数的多项式生成 ===")

    dataset = np.array([[1, 2]], dtype=float)
    print(f"输入数据: {dataset}")

    for degree in range(1, 4):
        result = generate_polynomials(dataset, degree)
        print(f"\n度数 {degree} 的结果:\n{result}")
        print(f"形状: {result.shape}")


# debug_generate_polynomials()
# debug_with_different_degrees()
