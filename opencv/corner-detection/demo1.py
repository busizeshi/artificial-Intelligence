import cv2
import numpy as np

# 读取图片
img_path = r'./test.png'  # 或者使用绝对路径
img = cv2.imread(img_path)

# 检查是否读取成功
if img is None:
    raise FileNotFoundError(f"无法读取图片，请检查路径: {img_path}")

# 调整大小
img_resized = cv2.resize(img, (640, 520))

# 转灰度
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi 角点检测
# 参数：灰度图, 最大角点数, 品质因子(0-1), 角点间最小欧式距离
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

# 绘制角点
if corners is not None:
    corners = np.int0(corners)  # 转整数坐标
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img_resized, (x, y), 5, (0, 255, 0), -1)

# 显示结果
cv2.imshow('Shi-Tomasi Corner Detection', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
