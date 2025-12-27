"""
轮廓检测
"""
import cv2
import numpy as np
import tools


def generate_dummy_image():
    """生成包含三角形、矩形、圆形的测试图"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # 画矩形
    cv2.rectangle(img, (30, 30), (130, 130), (255, 255, 255), -1)
    # 画圆形
    cv2.circle(img, (250, 80), 50, (255, 255, 255), -1)
    # 画三角形
    pts = np.array([[100, 250], [200, 350], [300, 250]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 255, 255))
    return img


# 获取图像
img = generate_dummy_image()
img_display = img.copy()

# 2. 预处理：灰度 -> 高斯模糊 -> 二值化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 模糊是为了去除噪点，防止检测出细小的假轮廓
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 二值化：因为背景黑物体白，直接简单阈值即可
_, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

# 3. 查找轮廓
# cv2.RETR_EXTERNAL: 我们只关心独立的物体，不关心内部是否有孔
# cv2.CHAIN_APPROX_SIMPLE: 节省内存，只存拐点
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"检测到了 {len(contours)} 个轮廓")

# 分析每一个轮廓
for cnt in contours:
    # --- A. 计算面积 ---
    area = cv2.contourArea(cnt) # 计算面积,实质上是像素点的多少
    # 过滤掉太小的噪点（生产中非常重要）
    if area < 500:
        continue

    # --- B. 计算周长 ---
    # True 表示轮廓是闭合的
    perimeter = cv2.arcLength(cnt, True) # 计算周长

    # --- C. 多边形拟合 (关键步骤) ---
    # 这一步是为了把由几百个像素点组成的“圆形”或“不规则图形”
    # 简化成几条线段组成的形状。
    # 0.02 * perimeter 是精度参数（epsilon），越小拟合越精确，越大越粗糙
    epsilon = 0.02 * perimeter
    """
    对轮廓进行多边形逼近（多边形拟合）
    cnt：待拟合的轮廓点集（由 cv2.findContours() 返回）
    epsilon：逼近精度参数，通常为轮廓周长的百分比（代码中为 0.02 * perimeter）
    True：布尔值，表示轮廓是否闭合（True 表示闭合轮廓）
    """
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # approx 里面存的是角点（顶点）
    corners = len(approx)
    shape_name = "Unknown"

    # 根据角点数量判断形状
    if corners == 3:
        shape_name = "Triangle"
    elif corners == 4:
        # 还可以通过长宽比判断是正方形还是长方形
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif corners > 4:
        # 角点很多，通常是圆形
        shape_name = "Circle"

    # --- D. 计算重心 (Moments) ---
    # 矩（Moments）可以计算图像的质心
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # --- E. 绘制结果 ---
    # 画出轮廓线
    cv2.drawContours(img_display, [cnt], -1, (0, 255, 0), 3)
    # 画出中心点
    cv2.circle(img_display, (cX, cY), 5, (0, 0, 255), -1)
    # 写上文字
    cv2.putText(img_display, f"{shape_name}", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

tools.show_images_with_titles([img, img_display], ['Original', 'Result'])
