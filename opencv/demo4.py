"""
边缘检测算法
"""
import cv2
import numpy as np
import tools

img=cv2.imread('../data/media/girl1.jpg',cv2.IMREAD_GRAYSCALE)

# 高斯平滑
img_blur=cv2.GaussianBlur(img,(5,5),1.4)

# canny边缘检测
edges=cv2.Canny(img_blur,50,150)

# 可选：膨胀边缘增强缺陷
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
edges_dilated = cv2.dilate(edges, kernel)

# 5. 标记缺陷区域
"""
findContours函数查找图像中的轮廓
1、在经过膨胀处理的边缘图像edges_dilated中检测轮廓
2、RETR_EXTERNAL参数表示只检测最外层轮廓
3、CHAIN_APPROX_SIMPLE参数表示使用简单逼近方法存储轮廓点
返回检测到的轮廓列表和层次结构信息
"""
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    if cv2.contourArea(cnt) > 50:  # 忽略小噪声
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0,0,255), 2)

tools.show_images_with_titles([img, edges, edges_dilated, output], ['Original', 'Canny', 'Dilated', 'Defects'])