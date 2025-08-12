from PIL import Image
import pytesseract
import cv2
import os

image = cv2.imread('images/scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 滤波或者二值化下，使得图片更加清楚
# thresh是阈值处理，blur是模糊处理
preprocess = 'thresh'
if preprocess == "thresh":
    # OTSU自动确定阈值
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    img = Image.open(filename)
    text = pytesseract.image_to_string(img)
    print(text)


    def cv_show(title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv_show('t', gray)