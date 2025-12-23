"""
图像基本操作
"""
import cv2
import numpy as np
from torch.nn.parallel import replicate

def img_show(window_name, img):
    cv2.imshow('girl1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
图像读取
"""
"""

img = cv2.imread('../data/media/girl1.jpg')
print(img.shape)
# (1439, 1080, 3)
# img_show('girl1', img)

img=cv2.imread('../data/media/girl1.jpg', cv2.IMREAD_GRAYSCALE)
# img_show('girl1', img)
print(type( img))
# <class 'numpy.ndarray'>
# 保存
# cv2.imwrite('../data/media/girl1_gray.jpg', img)
print(img.size)
"""

"""
视频读取
"""
"""
video=cv2.VideoCapture('../data/media/dog.mp4')
while True:
    flag, frame = video.read()
    if flag:
        cv2.imshow('video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()
"""

"""
截取部分图像数据
"""
# img=cv2.imread('../data/media/girl1.jpg')
# cat=img[0:500, 0:500]
# img_show('cat', cat)

"""
颜色通道提取
"""
# b,g,r=cv2.split(img)
# img_show('b', b)
# img_show('g', g)
# img_show('r', r)
# img=cv2.merge([r,g,b])

"""
边界填充
- BORDER_REPLICATE：复制法，也就是复制最边缘像素。
- BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb   
- BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
- BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg  
- BORDER_CONSTANT：常量法，常数值填充。
"""
# top_size,bottom_size,left_size,right_size=(50,50,50,50)
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# img_show('replicate', replicate)
# reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
# img_show('reflect', reflect)
# reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# img_show('reflect101', reflect101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
# img_show('wrap', wrap)
# constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
# img_show('constant', constant)

"""
数值计算
"""
img1=cv2.imread('../data/media/girl1.jpg')
img2=cv2.imread('../data/media/girl2.jpg')
"""
# 相当于每个像素都加10
超过255的自动取余
"""
# img11=img1+10
# img_show('img11', img11)

"""
图像融合
"""
# 打印两张图片的尺寸信息，确保它们具有兼容的尺寸才能进行混合操作
print(img1.shape)
print(img2.shape)

# 调整img2的尺寸以匹配img1的尺寸，确保两张图片大小相同才能进行混合操作
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 方法1: 直接相加图像
# 将两张图像像素值直接相加，超出255的部分会被截断或取模
img3 = img1 + img2
# img_show('img3', img3)

# 方法2: 使用cv2.addWeighted()进行加权混合 - 这是更常用的图像融合方法
# 这是OpenCV中专门用于图像融合的函数
# 语法: cv2.addWeighted(img1, alpha, img2, beta, gamma)
# img1: 第一张图像
# alpha: 第一张图像的权重 (0-1之间的值，这里为0.4)
# img2: 第二张图像
# beta: 第二张图像的权重 (0-1之间的值，这里为0.6)
# gamma: 亮度调节参数，通常设为0
# 计算公式: img4 = img1 * alpha + img2 * beta + gamma
# 这里计算为: img4 = img1 * 0.4 + img2 * 0.6 + 0
# 结果是第一张图像占40%，第二张图像占60%的混合效果
img4 = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
img_show('img4', img4)