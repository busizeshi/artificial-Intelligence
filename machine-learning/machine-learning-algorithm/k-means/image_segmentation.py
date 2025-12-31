import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# 1. 获取图像数据
original_img = cv2.imread('C:\\Users\\13127\\Desktop\\share\\imgs\\girl1.jpg')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转换BGR到RGB
h, w, c = original_img.shape

# 2. 数据转换
# 将 (height, width, channels) 转换为 (n_pixels, channels)
# 这样每个像素就变成了一个 3 维空间的坐标点 (R, G, B)
pixels = original_img.reshape((-1, 3))

# 3. 应用 K-Means 聚类
# 假设我们要将图像压缩成 3 种主色调
K = 3
print(f"正在对图像像素进行聚类 (K={K})...")
km = KMeans(n_clusters=K, random_state=42, n_init=10)
km.fit(pixels)

# 4. 重构图像
# 获取每个像素所属的簇标签
labels = km.labels_
# 获取每个簇的中心颜色（即该类的平均颜色）
centers = km.cluster_centers_.astype(np.uint8)

# 用对应的簇中心颜色替换原始像素颜色
segmented_pixels = centers[labels]
segmented_img = segmented_pixels.reshape((h, w, c))

# 5. 可视化对比
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title(f"Segmented Image (K={K})")
plt.axis('off')

plt.tight_layout()
plt.show()

print("图像分割任务已完成。")
