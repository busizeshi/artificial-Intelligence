import cv2
import numpy as np
import matplotlib.pyplot as plt

def plt_show_img(img):
    """
    显示图像的通用函数
    :param img: 待显示的图像
    """
    plt.imshow(img)
    plt.show()


def show_images_with_titles(images, titles, rows=1, cols=None):
    """
    显示多张图像的通用函数
    :param images: 图像列表
    :param titles: 每张图像对应的标题列表
    :param rows: 显示的行数
    :param cols: 显示的列数，默认为None，会根据图像数量和行数自动计算
    """
    # 如果未指定列数，则根据图像数量和行数计算
    if cols is None:
        cols = len(images) // rows if len(images) % rows == 0 else len(images) // rows + 1

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # 如果只有一张图，axes不会是数组，需要特殊处理
    if len(images) == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = axes  # 保持为数组
    elif rows > 1 and cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()  # 转为一维数组便于遍历

    # 遍历每张图像并显示
    for i in range(len(images)):
        # 检查图像是否为彩色图（3通道）
        if len(images[i].shape) == 3:
            # 将BGR格式转换为RGB格式
            rgb_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            axes[i].imshow(rgb_img)
        else:
            # 灰度图直接显示
            axes[i].imshow(images[i], cmap='gray')

        # 设置标题
        axes[i].set_title(titles[i])
        # 关闭坐标轴
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    # 调整子图间距
    plt.tight_layout()
    # 显示图像
    plt.show()
