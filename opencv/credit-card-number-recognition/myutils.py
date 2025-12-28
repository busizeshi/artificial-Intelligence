import cv2
import numpy as np


def sort_contours(cnts, method="left-to-right"):
    """
    对轮廓进行排序
    :param cnts: 轮廓列表
    :param method: 排序方式 "left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"
    :return: (排序后的轮廓, 对应的包围框)
    """
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # 0 是 x 坐标 (左右), 1 是 y 坐标 (上下)
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 计算每个轮廓的包围框
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # 使用 zip 将轮廓和包围框组合，根据包围框的 x 或 y 坐标排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    调整图像大小，保持纵横比
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def cv_show(name, img, wait=True):
    """
    显示图像的辅助函数
    :param wait: 是否阻塞等待按键，调试时很有用
    """
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()