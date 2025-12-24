"""
图像处理
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.gapi import kernel

import tools

"""
灰度图
"""
# img=cv2.imread('../data/opencv/girl2.jpg')
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)
# plt_show_img(img_gray)

"""
HSV
"""
# img=cv2.imread('../data/opencv/girl2.jpg')
# img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# h,s,v=cv2.split(img_hsv)

# 创建子图展示HSV三个通道
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# H:色调，表示颜色的基本类型（eg、红、黄、绿、蓝）
# axes[0].imshow(h, cmap='hsv')
# axes[0].set_title('Hue (H)')
# axes[0].axis('off')

# S:饱和度，表示颜色的纯度（eg、纯红、纯黄、纯绿、纯蓝）
# axes[1].imshow(s, cmap='gray')
# axes[1].set_title('Saturation (S)')
# axes[1].axis('off')

# V:明度，表示颜色的亮度（eg、最亮、最暗）
# axes[2].imshow(v, cmap='gray')
# axes[2].set_title('Value (V)')
# axes[2].axis('off')
#
# plt.tight_layout()
# plt.show()

#hsv提取蓝色物体
# img=cv2.imread('../data/opencv/girl3.jpg')
# 将BGR图像转换为HSV色彩空间
# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# 定义蓝色在HSV空间的范围
# H（色调）：蓝色的色调范围大约在100-130之间
# S（饱和度）：80-255，确保颜色足够饱和
# V（明度）：80-255，确保亮度适中
# lower_blue=np.array([100,80,80])
# upper_blue=np.array([130,255,255])

# 创建蓝色掩码
# cv2.inRange函数会返回在指定范围内的像素为白色（255），其他像素为黑色（0）
# mask=cv2.inRange(hsv,lower_blue,upper_blue)

# 应用掩码，提取蓝色物体
# cv2.bitwise_and函数将原图与自身进行按位与操作，只保留掩码中白色部分的像素
# result = cv2.bitwise_and(img, img, mask=mask)

# 使用新函数显示图像
# tools.show_images_with_titles([img, mask, result], ['Original Image', 'Mask', 'Result'], rows=1)

"""
图像阈值
"""
# img=cv2.imread('../data/opencv/girl2.jpg')

# 二值阈值
# ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# tools.show_images_with_titles([img,thresh1],['Original Image','Binary Threshold'])

# 反二值
# ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# tools.show_images_with_titles([img,thresh2],['Original Image','Binary Threshold Inverse'])

# 截断阈值
# ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# tools.show_images_with_titles([img,thresh3],['Original Image','Truncated Threshold'])

# ToZero
# ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# tools.show_images_with_titles([img,thresh4],['Original Image','ToZero Threshold'])

# 反ToZero
# ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# tools.show_images_with_titles([img,thresh5],['Original Image','ToZero Inverse Threshold'])

# 自适应阈值
# img=cv2.imread('../data/opencv/girl2.jpg',cv2.IMREAD_GRAYSCALE)
# thresh6=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# tools.show_images_with_titles([img,thresh6],['Original Image','Adaptive Threshold'])

# Otsu 阈值（自动阈值）
# img=cv2.imread('../data/opencv/girl2.jpg',cv2.IMREAD_GRAYSCALE)
# ret,thresh7=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# tools.show_images_with_titles([img,thresh7],['Original Image','Otsu Threshold'])

# img=cv2.imread('../data/opencv/girl2.jpg')
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 普通阈值
# _,thresh1=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# Otsu自动阈值
# _,thresh2=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# 自适应阈值
# adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# tools.show_images_with_titles([img,thresh1,thresh2,adaptive],['Original Image','Threshold','Otsu Threshold','Adaptive Threshold'])

"""
图像平滑(滤波算法)
"""
# img=cv2.imread('../data/opencv/dog1.png')

# 均值滤波
# mean_blur=cv2.blur(img,(5,5))
# 高斯滤波
# gaussian_blur=cv2.GaussianBlur(img,(5,5),0)
# 中值滤波
# median_blur=cv2.medianBlur(img,5)
# 双边滤波
# bilateral_blur=cv2.bilateralFilter(img,9,75,75)
# tools.show_images_with_titles(
#     [img,mean_blur,gaussian_blur,median_blur,bilateral_blur],
#     ['Original Image','Mean Blur','Gaussian Blur','Median Blur','Bilateral Blur'])


"""
腐蚀操作
"""
# img=cv2.imread('../data/opencv/dige.png')
# kernel=np.ones((5,5),np.uint8)  #结构元素
# erosion=cv2.erode(img,kernel,iterations=1)
# tools.show_images_with_titles([img,erosion],['Original Image','Erosion'])

"""
膨胀操作
"""
# img=cv2.imread('../data/opencv/dige.png')
# kernel=np.ones((5,5),np.uint8)
# dilation=cv2.dilate(img,kernel,iterations=1)
# tools.show_images_with_titles([img,dilation],['Original Image','Dilation'])

"""
开运算和闭运算
"""
# img=cv2.imread('../data/opencv/dige.png')
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# tools.show_images_with_titles([img,opening],['Original Image','Opening'])
# closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
# tools.show_images_with_titles([img,closing],['Original Image','Closing'])

"""
形态学梯度
"""
# img=cv2.imread('../data/opencv/girl1.jpg')
# 二值化
# _,binary=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# 结构元素
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# gradient=cv2.morphologyEx(binary,cv2.MORPH_GRADIENT,kernel)
# tools.show_images_with_titles([img,binary,gradient],['Original Image','Binary','Gradient'])

"""
礼帽和黑帽
"""
img=cv2.imread('../data/opencv/girl1.jpg')
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
# 礼帽
tophat=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
# 黑帽
blackhat=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
tools.show_images_with_titles([img,tophat,blackhat],['Original Image','Top Hat','Black Hat'])