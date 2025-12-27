"""
直方图均衡化
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
# 为了效果明显，最好找一张雾蒙蒙的或者低对比度的灰度图
# 如果没有，我们手动生成一张低对比度的图
img = cv2.imread("D:\\resource\\girl.jpg", 0)

if img is None:
    print("未找到图片，生成模拟低对比度图像...")
    # 生成一张值集中在 100-150 之间的灰色图
    img = np.random.normal(125, 10, (512, 512)).astype(np.uint8)
    # 加一点图案方便观察
    cv2.putText(img, "OPENCV", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, 140, 5)
    cv2.circle(img, (400, 100), 50, 110, -1)

# ==========================================
# 方法 A: 全局直方图均衡化 (Global HE)
# ==========================================
# 简单粗暴，但容易产生噪点或过度亮化
equ = cv2.equalizeHist(img)

# ==========================================
# 方法 B: CLAHE (自适应直方图均衡化) - 推荐！
# ==========================================
# clipLimit: 颜色对比度的阈值，超过这个值会被剪裁并均匀分布到其他直方图柱子上
# tileGridSize: 图像被分成多少块进行局部处理 (8x8 是常用值)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)

# ==========================================
# 绘制结果与直方图对比
# ==========================================
plt.figure(figsize=(12, 8))

# 1. 原图
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original (Low Contrast)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.hist(img.ravel(), 256, [0, 256], color='gray')
plt.title('Original Histogram')
plt.xlim([0, 256])

# 2. 全局均衡化
plt.subplot(2, 3, 2)
plt.imshow(equ, cmap='gray')
plt.title('Global Equalization')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.hist(equ.ravel(), 256, [0, 256], color='gray')
plt.title('Global HE Histogram')
plt.xlim([0, 256])

# 3. CLAHE
plt.subplot(2, 3, 3)
plt.imshow(res_clahe, cmap='gray')
plt.title('CLAHE (Adaptive)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.hist(res_clahe.ravel(), 256, [0, 256], color='gray')
plt.title('CLAHE Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()