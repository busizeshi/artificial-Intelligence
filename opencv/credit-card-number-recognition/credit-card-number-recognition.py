import cv2
import numpy as np
import argparse
import myutils


class CreditCardOCR:
    def __init__(self, template_path, image_path, debug=False):
        self.template_path = template_path
        self.image_path = image_path
        self.debug = debug
        self.digits_template = {}

        # 定义信用卡类型前缀
        self.FIRST_NUMBER = {
            "3": "American Express",
            "4": "Visa",
            "5": "MasterCard",
            "6": "Discover Card"
        }

    def process_template(self):
        """
        步骤1: 处理模板图像，提取数字 0-9 的轮廓
        """
        img = cv2.imread(self.template_path)
        if img is None:
            raise FileNotFoundError(f"无法读取模板图像: {self.template_path}")

        # 显示原始模板
        myutils.cv_show("Template Original", img)

        # 预处理：灰度 -> 二值化
        ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

        # 显示处理后的模板（二值化）
        myutils.cv_show("Template Ref (Binary)", ref)

        # 查找轮廓
        refCnts, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)

        # 显示绘制了轮廓的模板
        myutils.cv_show("Template Contours", img)

        # 排序：从左到右 (0, 1, 2, ... 9)
        refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]

        # 遍历并将每个数字的ROI(Region of Interest)保存到字典
        for (i, c) in enumerate(refCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))  # 固定大小以便后续匹配
            self.digits_template[i] = roi

    def preprocess_card_image(self):
        """
        步骤2: 读取并预处理信用卡图像，通过形态学操作突出数字区域
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取输入图像: {self.image_path}")

        image = myutils.resize(image, width=300)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 显示调整大小后的原图和灰度图
        myutils.cv_show("Card Original", image)
        myutils.cv_show("Card Gray", gray)

        # 初始化卷积核
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

        # 礼帽操作：突出比背景亮的区域（数字）
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
        myutils.cv_show("Card Tophat", tophat)

        # Sobel算子计算x方向梯度，检测垂直边缘
        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")
        myutils.cv_show("Card Gradient (Sobel)", gradX)

        # 闭操作：将数字连接成块
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        myutils.cv_show("Card Gradient Close", gradX)

        # 二值化 (Otsu算法自动寻找阈值)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        myutils.cv_show("Card Threshold", thresh)

        # 二次闭操作，填充空洞
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        myutils.cv_show("Card Threshold Close (Final)", thresh)

        return image, gray, thresh

    def find_digit_groups(self, thresh):
        """
        步骤3: 从预处理后的图像中找到包含4个数字的组
        """
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        locs = []

        # 筛选符合信用卡数字组特征的轮廓
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # 经验参数：标准信用卡数字组（4位数字）的长宽比通常在 2.5 到 4.0 之间
            if 2.5 < ar < 4.0:
                if (40 < w < 55) and (10 < h < 20):
                    locs.append((x, y, w, h))

        # 将筛选出的组按从左到右排序
        locs = sorted(locs, key=lambda x: x[0])
        return locs

    def recognize_digits(self, image, gray, group_locs):
        """
        步骤4: 遍历每个数字组，分割单字并与模板匹配
        """
        output = []

        for (i, (gX, gY, gW, gH)) in enumerate(group_locs):
            groupOutput = []

            # 提取当前组的ROI，稍微扩大一点范围以防切断边缘
            group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]

            # 对组进行二值化，分离单个数字
            group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # 显示当前处理的数字组
            myutils.cv_show(f"Digit Group {i + 1}", group)

            # 检测组内的单个数字轮廓
            digitCnts, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = myutils.sort_contours(digitCnts, method="left-to-right")[0]

            for c in digitCnts:
                # 提取单个数字的ROI
                (x, y, w, h) = cv2.boundingRect(c)
                roi = group[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))  # 必须与模板大小一致

                # 显示单个数字 ROI (如果觉得弹窗太多可以注释掉这一行)
                # myutils.cv_show("Single Digit ROI", roi)

                # 模板匹配
                scores = []
                for (digit, digitROI) in self.digits_template.items():
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                # 取分数最高的数字
                groupOutput.append(str(np.argmax(scores)))

            # 在原图上绘制矩形和识别结果
            cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
            cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            output.extend(groupOutput)

        return output

    def run(self):
        """
        执行完整流程
        """
        print("[INFO] 正在处理模板...")
        self.process_template()

        print("[INFO] 正在处理信用卡图像...")
        image, gray, thresh = self.preprocess_card_image()

        print("[INFO] 正在定位数字区域...")
        group_locs = self.find_digit_groups(thresh)

        print(f"[INFO] 找到 {len(group_locs)} 组数字区域")
        if len(group_locs) == 0:
            print("[WARN] 未检测到符合条件的数字组，请检查图片清晰度或调整过滤参数。")
            return

        print("[INFO] 正在识别数字...")
        output = self.recognize_digits(image, gray, group_locs)

        print("-" * 30)
        print(f"检测结果: {''.join(output)}")
        if len(output) > 0 and output[0] in self.FIRST_NUMBER:
            print(f"卡片类型: {self.FIRST_NUMBER[output[0]]}")
        print("-" * 30)

        myutils.cv_show("Final Result", image)


if __name__ == "__main__":
    # 参数解析
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="输入信用卡图片的路径")
    ap.add_argument("-t", "--template", required=True, help="OCR-A 模板图片的路径")
    ap.add_argument("-d", "--debug", action="store_true", help="[已停用] 默认显示所有调试图片")
    args = vars(ap.parse_args())

    # 实例化并运行
    try:
        ocr = CreditCardOCR(args["template"], args["image"], args["debug"])
        ocr.run()
    except Exception as e:
        print(f"[ERROR] 发生错误: {e}")