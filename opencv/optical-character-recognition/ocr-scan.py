import cv2
import numpy as np
import argparse
import pytesseract


def resize(image, height=None, width=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if height is None and width is None:
        return image

    if height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def find_document_contour(edged):
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

    return None


def preprocess(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    if debug:
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edged


def scan_document(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("无法读取图片")

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = resize(image, height=500)

    edged = preprocess(image, debug)
    screenCnt = find_document_contour(edged)

    if screenCnt is None:
        raise RuntimeError("未检测到文档轮廓")

    if debug:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def ocr_image(image):
    """
    对扫描后的图像进行 OCR（中英文）
    """
    text = pytesseract.image_to_string(
        image,
        # lang="chi_sim+eng",
        # config="--psm 6"
    )
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    ap.add_argument("--debug", action="store_true", help="Show debug windows")
    args = ap.parse_args()

    # 1. 文档扫描
    scanned = scan_document(args.image, debug=args.debug)
    cv2.imwrite("scan.jpg", scanned)

    # 2. OCR 识别
    text = ocr_image(scanned)
    print("===== OCR 结果 =====")
    print(text)

    # 3. 可视化扫描结果
    cv2.imshow("Scanned", resize(scanned, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()