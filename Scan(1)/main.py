import numpy as np
import cv2
import os
from utils import resize, cv_show, four_point_transform, get_screenCnt, get_rotated_img, ocr_check
from length import calculateWH
from edgeseek import DocScanner
from Clip import Clip

# 读取图片
image = cv2.imread("images/test5.jpg")


def img_ocr(image):
    # 直接传入图像数组
    clip = Clip(image)
    cropped_image = clip.run()

    if cropped_image is not None:
        output_path = os.path.abspath("images/ropped_result.jpg")
        cv2.imwrite(output_path, cropped_image)
        print(f"结果已保存到: {output_path}")


    h, w = cropped_image.shape[0:2]
    gray = cv2.medianBlur(cropped_image, 3)
    orig = cropped_image.copy()
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
    cv2.imwrite("images/warped.jpg",warped)



    # 二值处理
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 灰度图像直方图均衡化
    alpha = 1.5  # 对比度系数

    beta = -50  # 亮度调整
    dst = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)

    threshold_value = 240
    ref = np.where(dst > threshold_value, 255, dst)


    # ref = cv2.threshold(img_processed, 150, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('result/AW.jpg',ref)
    print(int(calculateWH(screenCnt)[1]))
    resize_img = resize(ref, height=int(calculateWH(screenCnt)[1])*2, width=int(calculateWH(screenCnt)[0])*2)
    # 保存到文件
    cv2.imwrite('result/scan5.jpg', resize_img)


text = img_ocr(image)
