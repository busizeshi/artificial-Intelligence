import cv2
import numpy as np
import tools

"""
高斯金字塔
"""
# img= cv2.imread("D:\\resource\\girl.jpg")
# gp=[img]
# for i in range(4):
#     img=cv2.pyrDown(img)
#     gp.append(img)
# tools.show_images_with_titles([img for img in gp], ['G0', 'G1', 'G2', 'G3', 'G4'])

"""
拉普拉斯金字塔
"""
src = cv2.imread("D:\\resource\\girl.jpg")

# 构建高斯金字塔
layer = src.copy()
gaussian_pyramid = [layer]
for i in range(3):
    layer = cv2.pyrDown(layer)
    gaussian_pyramid.append(layer)

# 构建拉普拉斯金字塔
laplacian_pyramid = [gaussian_pyramid[3]]  # 塔顶（最小的高斯图）直接存入
for i in range(3, 0, -1):
    gaussian_current = gaussian_pyramid[i - 1]
    gaussian_upper = gaussian_pyramid[i]

    # 上采样
    gaussian_upper_upsampled = cv2.pyrUp(gaussian_upper)
    # 尺寸对齐：因为pyrDown可能会丢弃奇数行 / 列，pyrUp回来尺寸可能不一致
    h, w, _ = gaussian_current.shape
    gaussian_upper_upsampled = cv2.resize(gaussian_upper_upsampled, (w, h))
    # 计算差值 (Laplacian = Gaussian_i - Up(Gaussian_i+1))
    laplacian = cv2.subtract(gaussian_current, gaussian_upper_upsampled)
    laplacian_pyramid.append(laplacian)

# 修改拉普拉斯层（增强细节！）
# laplacian_pyramid 现在的顺序是 [Top(small), L2, L1, L0(big)]
# 我们增强 L0 (最底层的细节，包含最高频的纹理)
# 将细节乘以一个系数，比如 2.0，甚至 3.0
enhanced_laplacian_pyramid = []
for i, lap in enumerate(laplacian_pyramid):
    if i == 3:  # 也就是 L0，最底层的细节
        # 放大细节数值
        # 注意：需要转成 float 计算防止溢出，然后再转回 uint8
        lap_float = lap.astype(np.float32) * 3.0
        lap_enhanced = np.clip(lap_float, 0, 255).astype(np.uint8)
        enhanced_laplacian_pyramid.append(lap_enhanced)
    else:
        enhanced_laplacian_pyramid.append(lap)
# ==========================================
# 从拉普拉斯金字塔重建图像
# ==========================================
# 重建逻辑：G_i = L_i + Up(G_i+1)
reconstructed_img = enhanced_laplacian_pyramid[0]
for i in range(1, 4):
    laplacian = enhanced_laplacian_pyramid[i]
    reconstructed_img = cv2.pyrUp(reconstructed_img)

    # 尺寸对齐
    h, w, _ = laplacian.shape
    reconstructed_img = cv2.resize(reconstructed_img, (w, h))

    reconstructed_img = cv2.add(reconstructed_img, laplacian)

tools.show_images_with_titles([src, reconstructed_img], ['Original', 'Reconstructed'])
