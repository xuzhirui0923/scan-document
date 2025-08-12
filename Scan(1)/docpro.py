import numpy as np
import cv2
def cv_show(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        """resize函数之所以自定义，是可以只指定高度或者高度
            原理就是：
            如果只指定某一个维度，图片的高度和宽度都会同比例缩小，比如指定height，那就宽度变成height/float(h)*w, 高度为height， 指定width同理
            如果都指定， 那么就按照实际的大小resize
        """
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    #读取图片
    image = cv2.imread("images/receipt.jpg")
    cv_show('img', image)  # 原始图片是2448*3264, 太大了，下面需要进行resize操作
    image.shape

    # resize操作之前，需要保存resize的比例，以及原始图像
    # 下面要按照500的比例对图像进行resize， 那么原始图像的每个像素点的位置都会改变，记住ratio是为了最终能还原到原始的位置上去
    ratio = image.shape[0] / 500.0
    orig = image.copy()

    image = resize(orig, height=500)
    cv_show('img', image)

    # 预处理 转成灰度图 -> 高斯滤波 -> Canny边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)   边缘检测算法其实就是用的高斯滤波，所以这里这个不用发现更加清晰些
    edged = cv2.Canny(gray, 75, 200)

    cv_show('img', image)
    cv_show('edge', edged)

    # 轮廓检测
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # 下面要获取到最外围的大轮廓， 因为我们只需要这个大轮廓里面的所有东西， 外面黑色的背景其实不需要
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # C表示输入的点集
        # epsilon表示从原始轮廓到近似轮廓的最大距离
        # True表示封闭
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4个点的时候，说明是最外面的大轮廓，此时把这个拿出来
        if len(approx) == 4:
            screenCnt = approx
            break
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
            cv_show("Outline", image)

            def order_points(pts):
                # 一共4个坐标点
                rect = np.zeros((4, 2), dtype="float32")

                # 下面这个操作，是因为这四个点目前是乱序的，下面通过了一种巧妙的方式来找到对应位置
                # 左上和右下， 对于左上的这个点，(x,y)坐标和会最小， 对于右下这个点，(x,y)坐标和会最大，所以坐标求和，然后找最小和最大位置就是了
                # 按照顺序找到对应坐标0123分别是左上， 右上， 右下，左下
                s = pts.sum(axis=1)
                # 拿到左上和右下
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                # 右上和左下， 对于右上这个点，(x,y)坐标差会最小，因为x很大，y很小， 而左下这个点， x很小，y很大，所以坐标差会很大
                # 拿到右上和左下
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                return rect

            def four_point_transform(image, pts):
                # 拿到正确的左上，右上， 右下，左下四个坐标点的位置
                rect = order_points(pts)
                (tl, tr, br, bl) = rect

                # 计算输入的w和h值 这里就是宽度和高度，计算方式就是欧几里得距离，坐标对应位置相减平方和开根号
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                # 变换后对应坐标位置
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                # 计算变换矩阵  变换矩阵这里，要有原始的图像的四个点的坐标， 变换之后的四个点的对应坐标，然后求一个线性矩阵，相当于每个点通过一个线性映射
                # 到新的图片里面去。那么怎么求线性矩阵呢？
                # 其实解线性方程组， 原始图片四个点坐标矩阵A(4, 2)， 新图片四个点坐标矩阵B(4, 2)， 在列这个维度上扩充1维1
                # A变成了(4, 3), B也是(4, 3)， 每个点相当于(x, y, 1)
                # B = WA， 其中W是3*3的矩阵，A的每个点是3*1， B的每个点是3*1
                # W矩阵初始化[[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]]  这里面8个未知数，通过上面给出的4个点
                # 所以这里A， B四个点的坐标都扩充了1列，已知A,B四个点的坐标，这里去求参数，解8个线性方程组得到W，就是cv2.getPerspectiveTransform干的事情
                # 这个文章说的不错：https://blog.csdn.net/overflow_1/article/details/80330835
                W = cv2.getPerspectiveTransform(rect, dst)
                # 有了透视矩阵W, 对于原始图片中的每个坐标， 都扩充1列，然后与W乘， 就得到了在变换之后图片的坐标点(x, y, z), 然后把第三列给去掉(x/z, y/z)就是最终的坐标
                warped = cv2.warpPerspective(image, W, (maxWidth, maxHeight))

                # 返回变换后结果
                return warped

            cv_show('img', image)

            # 下面需要做透视变换， 让整个框里面的对象铺满整个图片，其余的地方去掉
            # 这个函数传入的是原始图片(resize之前的那个), resize之后的大轮廓坐标点乘以ratio，此时 轮廓坐标在resize之前图片里面的值
            # 这里是这样的， resize操作之后，其实原始图片的每个点的坐标都相对于原来点的坐标变小了，于是乎需要乘以这个ratio，才是在原始图片里面的坐标
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            cv_show('warped', warped)
            # 二值处理
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            ref = cv2.threshold(warped, 150, 255, cv2.THRESH_BINARY)[1]
            cv_show('ref', ref)

            rows, cols = ref.shape[:2]
            center = (cols / 2, rows / 2)  # 以图像中心为旋转中心
            angle = 90  # 顺时针旋转90°
            scale = 1  # 等比例旋转，即旋转后尺度不变

            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_img = cv2.warpAffine(ref, M, (cols, rows))
            cv_show('scan_img', rotated_img)
            resize_img = resize(rotated_img, height=600)
            cv_show('resize_img', resize_img)
            # 保存到文件
            cv2.imwrite('images/scan.jpg', rotated_img)