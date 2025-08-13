import numpy as np
import cv2


class DocScanner(object):

    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE

    def get_contour(self, rescaled_image):
        """核心函数：获取文档轮廓"""
        # 图像预处理
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 边缘检测
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        edged = cv2.Canny(dilated, 0, 84)

        # 查找轮廓
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # 寻找最佳四边形轮廓
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        # 如果没找到，返回整个图像边界
        h, w = rescaled_image.shape[:2]
        return np.array([[w, 0], [w, h], [0, h], [0, 0]], dtype="float32")