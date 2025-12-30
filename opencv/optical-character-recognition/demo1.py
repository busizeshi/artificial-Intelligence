import cv2
import pytesseract

img=cv2.imread('../../data/opencv/ocr1.png')
if img is None:
    raise FileNotFoundError('未找到图片')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
text=pytesseract.image_to_string(blur,lang='chi_sim+eng',config='--psm 6')
print( text)