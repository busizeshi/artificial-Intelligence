import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureMatcher:
    def __init__(self, method='SIFT', flann=True):
        """
        初始化匹配器类

        参数:
        :param method: 'SIFT' (精度高，推荐用于人物、复杂物体) 或 'ORB' (速度快，适合刚性物体)
        :param flann: 是否使用 FLANN 快速近似最近邻搜索 (True/False)，False则使用暴力匹配
        """
        self.method = method
        self.flann = flann

        # 1. 初始化特征检测器
        if method == 'SIFT':
            # SIFT 对尺度变化和旋转非常鲁棒，适合人物匹配
            self.detector = cv2.SIFT_create()

            # FLANN 参数配置 (对于 SIFT，使用 K-D Tree 算法)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # 搜索次数，越高精度越高但越慢

            # 初始化匹配器
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params) if flann else cv2.BFMatcher()

        elif method == 'ORB':
            # ORB 速度极快，适合实时系统，但对模糊和非刚性变形抵抗力较弱
            self.detector = cv2.ORB_create(nfeatures=2000)  # 增加特征点数量以提高召回率

            # FLANN 参数配置 (对于 ORB，使用 LSH 局部敏感哈希算法)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)

            # ORB 是二进制描述符，需使用 Hamming 距离
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params) if flann else cv2.BFMatcher(
                cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError("Method must be 'SIFT' or 'ORB'")

    def preprocess(self, img):
        """
        图像预处理：灰度化 + CLAHE (自适应直方图均衡化)
        作用：增强图像对比度，使暗处的特征点更容易被提取出来。
        """
        # 如果是彩色图，转为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 应用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 这对光照变化大的人物匹配非常有效，能提取更多纹理细节
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def match(self, img_query, img_train, ratio_thresh=0.75, min_match_count=10):
        """
        执行匹配流程并返回纵向拼接的结果图

        参数:
        :param img_query: 待寻找的目标图像 (如人物剪切图，模板)
        :param img_train: 包含目标的场景大图 (搜索区域)
        :param ratio_thresh: Lowe's ratio test 阈值 (通常 0.7 - 0.8)，越小越严格
        :param min_match_count: 要求的最小匹配点数，少于此数视为匹配失败
        """
        # 1. 预处理 (转灰度 + 增强对比度)
        gray_query = self.preprocess(img_query)
        gray_train = self.preprocess(img_train)

        # 2. 检测特征点(Keypoints)和计算描述符(Descriptors)
        kp1, des1 = self.detector.detectAndCompute(gray_query, None)
        kp2, des2 = self.detector.detectAndCompute(gray_train, None)

        # 检查是否提取到了特征
        if des1 is None or des2 is None:
            print("警告: 未检测到足够的特征点")
            return None, 0

        # 3. KNN 匹配 (K-Nearest Neighbors, k=2)
        # 为每个 query点 找出 train图中 最相似的 2 个点
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # 4. Lowe's Ratio Test (比率测试 - 关键过滤步骤)
        # 如果 最近邻距离 < 0.75 * 次近邻距离，说明匹配很独特，保留该点
        # 否则说明该特征太普通（有歧义），丢弃
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        print(f"原始匹配数: {len(matches)}, 经过比率测试后的匹配数: {len(good_matches)}")

        # 5. RANSAC 几何验证 (剔除外点 - 核心优化步骤)
        # 只有当匹配点构成合理的几何变换（如透视变换）时，才认为是有效匹配
        final_matches_mask = None  # 用于标记哪些点是内点(1)哪些是外点(0)
        homography = None
        img_train_vis = img_train.copy()  # 用于画框的场景图副本

        if len(good_matches) > min_match_count:
            # 获取匹配点对的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算单应性矩阵 (Homography) 并使用 RANSAC 剔除错误匹配
            # ransacReprojThreshold=5.0 表示允许 5 像素的投影误差
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                final_matches_mask = mask.ravel().tolist()

                # 计算 RANSAC 后的有效内点数量
                inliers_count = np.sum(final_matches_mask)
                print(f"RANSAC 验证后的内点(Inliers)数量: {inliers_count}")

                # 6. 在场景图中画出目标边框 (验证几何位置)
                if homography is not None:
                    h, w = gray_query.shape
                    # 定义模板图的四个角点
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    # 将角点投影到场景图中
                    dst = cv2.perspectiveTransform(pts, homography)
                    # 画出绿色边框
                    img_train_vis = cv2.polylines(img_train_vis, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                if inliers_count < min_match_count * 0.6:
                    print("警告: RANSAC 内点过少，匹配结果可能不可靠")
            else:
                print("无法计算单应性矩阵")
        else:
            print(f"匹配点不足 - 需要至少 {min_match_count} 个点，实际只有 {len(good_matches)} 个")
            final_matches_mask = None

        # 7. 绘图返回 (自定义：纵向拼接)
        # OpenCV 的 drawMatches 默认是横向的，这里我们手动实现纵向拼接

        # 确保图片是 BGR 格式 (如果是灰度图转为彩色，以便画彩色线)
        img1_color = img_query if len(img_query.shape) == 3 else cv2.cvtColor(img_query, cv2.COLOR_GRAY2BGR)
        img2_color = img_train_vis if len(img_train_vis.shape) == 3 else cv2.cvtColor(img_train_vis, cv2.COLOR_GRAY2BGR)

        # 获取尺寸
        h1, w1 = img1_color.shape[:2]  # 模板图高度
        h2, w2 = img2_color.shape[:2]  # 场景图高度

        # 创建大画布：宽度取最大值，高度相加
        vis_h = h1 + h2
        vis_w = max(w1, w2)
        vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

        # 将图片放入画布
        # 上方：模板图
        vis[:h1, :w1] = img1_color
        # 下方：场景图
        vis[h1:h1 + h2, :w2] = img2_color

        # 绘制匹配线
        for i, m in enumerate(good_matches):
            # 如果 RANSAC 标记为外点(0)，则跳过不画
            if final_matches_mask is not None and final_matches_mask[i] == 0:
                continue

            # 获取坐标
            # pt1: 模板图中的特征点 (queryIdx)
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))

            # pt2: 场景图中的特征点 (trainIdx)
            # 关键：因为场景图放在下方，所以 Y 坐标要加上 h1 (模板图的高度)
            pt2_raw = kp2[m.trainIdx].pt
            pt2 = (int(pt2_raw[0]), int(pt2_raw[1] + h1))

            # 绘制线条 (绿色)
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            # 绘制端点 (可选：起点蓝色，终点红色)
            cv2.circle(vis, pt1, 3, (255, 0, 0), 1)
            cv2.circle(vis, pt2, 3, (0, 0, 255), 1)

        return vis, len(good_matches)


# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟数据生成 (实际使用时请替换为 cv2.imread('path.jpg'))
    # 创建一个简单的“场景”和“模板”
    scene = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(scene, (100, 100), (250, 300), (200, 200, 200), -1)
    cv2.circle(scene, (175, 150), 30, (150, 150, 255), -1)
    cv2.rectangle(scene, (150, 180), (200, 280), (255, 100, 100), -1)
    # 添加噪声模拟真实环境
    noise = np.random.randint(0, 50, (400, 400, 3)).astype(np.uint8)
    scene = cv2.add(scene, noise)

    # 模板就是场景的一部分
    template = scene[100:300, 100:250].copy()

    # 初始化匹配器 (推荐使用 SIFT)
    matcher = FeatureMatcher(method='SIFT')

    # 执行匹配
    result_img, count = matcher.match(template, scene)

    if result_img is not None:
        # 使用 matplotlib 显示结果
        plt.figure(figsize=(8, 12))  # 调整显示比例适应纵向图
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Vertical Stitching Result (Inliers: {count})")
        plt.axis('off')
        plt.show()