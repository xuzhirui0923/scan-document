import numpy as np
import cv2
import os
from utils import resize, cv_show, four_point_transform, get_screenCnt, get_rotated_img, ocr_check
from length import calculateWH
from edgeseek import DocScanner

# 读取图片
image = cv2.imread('images/page.jpg')


def img_ocr(image):
    h, w = image.shape[0:2]
    gray = cv2.medianBlur(image, 3)
    orig = image.copy()
    image = resize(orig, height=h)
    ratio = orig.shape[0] / float(image.shape[0])

    # 使用DocScanner类进行边缘检测和轮廓查找
    scanner = DocScanner(interactive=False)
    screenCnt = scanner.get_contour(image)

    # 确保screenCnt是正确格式的numpy数组
    if screenCnt is None or len(screenCnt) != 4:
        print("未找到有效的文档轮廓，使用全图")
        screenCnt = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype="int32")
    else:
        screenCnt = screenCnt.reshape(4, 1, 2).astype("int32")

    # 绘制最终选择的轮廓
    debug_img = image.copy()
    cv2.drawContours(debug_img, [screenCnt], -1, (0, 0, 255), 2)
    cv2.imwrite('result/Selected Contour.jpg', debug_img)

    # 透视变换
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # 二值处理
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('AW.jpg',ref)
    print(int(calculateWH(screenCnt)[1]))
    resize_img = resize(ref, height=int(calculateWH(screenCnt)[1])*2, width=int(calculateWH(screenCnt)[0])*2)
    # 保存到文件
    cv2.imwrite('scan1.jpg', resize_img)


text = img_ocr(image)
