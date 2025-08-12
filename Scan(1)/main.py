import numpy as np
import cv2
import os
from utils import resize, cv_show, four_point_transform, get_screenCnt, get_rotated_img, ocr_check
from length import calculateWH

# 读取图片
image = cv2.imread('images/page.jpg')


def img_ocr(image):
    h, w = image.shape[0:2]
    gray = cv2.medianBlur(image, 3)
    orig = image.copy()
    image = resize(orig, height=h)
    ratio = orig.shape[0] / float(image.shape[0])  # 计算实际比例

    # 预处理 转成灰度图 -> 高斯滤波 -> Canny边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)   边缘检测算法其实就是用的高斯滤波，所以这里这个不用发现更加清晰些
    edged = cv2.Canny(gray, 75, 200)

    # 闭运算滤波
    kernel = np.ones((2, 2), np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 保存边缘检测结果
    cv2.imwrite("result/edge.jpg", edged)

    # 轮廓检测
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = get_screenCnt(cnts)

    # 绘制所有候选轮廓
    debug_img = image.copy()
    cv2.drawContours(debug_img, cnts, -1, (0, 255, 0), 2)

    # 保存所有轮廓
    cv2.imwrite('result/All Contours.jpg', debug_img)

    # 绘制最终选择的轮廓
    debug_img = image.copy()
    if screenCnt is not None:
        cv2.drawContours(debug_img, [screenCnt], -1, (0, 0, 255), 2)
    cv2.imwrite('result/Selected Contour.jpg', debug_img)

    # 透视变换
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # cv_show('After Warp', warped)  # 检查透视变换结果
    # 二值处理
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]
    # cv_show('After Threshold', ref)  # 检查二值化效果
    cv2.imwrite('AW.jpg',ref)
    # rotated_img = get_rotated_img(ref)
    # cv2.imwrite('ro.jpg',rotated_img)
    print(int(calculateWH(screenCnt)[1]))
    resize_img = resize(ref, height=int(calculateWH(screenCnt)[1]), width=int(calculateWH(screenCnt)[0]))
    # 保存到文件
    cv2.imwrite('scan1.jpg', resize_img)


text = img_ocr(image)
