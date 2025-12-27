import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像 (必须是灰度图)
img = cv2.imread("D:\\resource\\girl.jpg", 0)

if img is None:
    # 没图就生成个简单的
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 400), 255, -1)
else:
    img = cv2.resize(img, (512, 512))

# ==========================================
# 第一步：傅里叶变换 (DFT)
# ==========================================
# dft 需要 float32 输入
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# 将低频分量移动到中心
dft_shift = np.fft.fftshift(dft)

# 计算幅度谱（用于显示，让人眼能看到频域长啥样）
# magnitude = 20*log(abs(f))，用对数尺度是因为中心的值往往太大了
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# ==========================================
# 第二步：设计滤波器 (高通滤波器 HPF)
# ==========================================
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2 # 中心点

# 创建一个掩模 mask，中心为 0 (阻挡)，四周为 1 (通过)
# 也就是把低频信息（平坦区域）扣掉
mask = np.ones((rows, cols, 2), np.uint8)
r = 30 # 滤波半径，半径越小，保留的低频越少，细节越多
# 将中心 r 半径内的区域设为 0
mask[crow-r:crow+r, ccol-r:ccol+r] = 0

# ==========================================
# 第三步：应用滤波器并逆变换 (IDFT)
# ==========================================
# 将 DFT 结果与 Mask 相乘
fshift = dft_shift * mask

# 将中心移回左上角
f_ishift = np.fft.ifftshift(fshift)

# 逆傅里叶变换
img_back = cv2.idft(f_ishift)

# 计算复数的幅度作为最终图像
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# ==========================================
# 可视化
# ==========================================
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("High Pass Filter Result")
# img_back 也是 float 类型，且范围可能很大，显示时最好归一化
plt.imshow(img_back, cmap='gray')
plt.axis('off')

plt.show()