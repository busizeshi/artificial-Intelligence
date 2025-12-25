"""
图像梯度
"""
import cv2
import tools
import numpy as np

"""
sobel算子
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
# img = cv2.imread('../data/media/girl1.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1️⃣ 计算 Sobel X 和 Y
# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 2️⃣ 计算梯度幅值
# sobel_mag = np.sqrt(sobelx**2 + sobely**2)
# sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

# 3️⃣ 阈值处理（边缘提取）
# _, sobel_edge = cv2.threshold(sobel_mag, 100, 255, cv2.THRESH_BINARY)

# 4️⃣ 显示
# tools.show_images_with_titles([img, sobel_edge], ['Original Image', 'Sobel Edge'])

"""
scharr算子
"""
# img = cv2.imread('../data/media/girl1.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
# scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

# 梯度幅值
# scharr_mag = np.sqrt(scharrx ** 2 + scharry ** 2)
# scharr_mag = np.uint8(np.clip(scharr_mag, 0, 255))

# 阈值处理
# _, scharr_edge = cv2.threshold(scharr_mag, 100, 255, cv2.THRESH_BINARY)
# tools.show_images_with_titles([img, scharr_edge], ['Original Image', 'Scharr Edge'])

"""
Laplacian算子
"""
# img = cv2.imread('../data/media/girl1.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯平滑降噪
# gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# 计算Laplacian
# laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F, ksize=3)

# 转为8位图显示
# laplacian_abs = cv2.convertScaleAbs(laplacian)

# 阈值处理
# _, laplacian_edge = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)
# tools.show_images_with_titles([img, laplacian_edge], ['Original Image', 'Laplacian Edge'])

"""
Canny
"""
img=cv2.imread('../data/media/girl1.jpg')
blur=cv2.GaussianBlur(img,(5,5),1.4)
canny=cv2.Canny(blur,50,150)
tools.show_images_with_titles([img,canny],['Original Image','Canny Edge'])