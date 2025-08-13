import cv2
import numpy as np
import screeninfo


class Clip:
    def __init__(self, image_input):
        """初始化裁剪器，支持文件路径或numpy数组"""
        self.image = cv2.imread(image_input) if isinstance(image_input, str) else image_input.copy()
        if self.image is None:
            raise ValueError("无法加载图像，请检查路径或输入数据")

        self.points = []  # 存储四边形顶点坐标
        self.selected_point = None  # 当前选中的点索引
        self.scale_factor = 1.0  # 图像显示缩放比例
        self.window_name = "Perspective Clipper"

        # 初始化
        self._setup_window()
        self._init_points()
        self._update_display()

    def _setup_window(self):
        """配置显示窗口"""
        self._calculate_scale_factor()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.scale_factor < 1.0:
            h, w = self.image.shape[:2]
            cv2.resizeWindow(self.window_name,
                             int(w * self.scale_factor),
                             int(h * self.scale_factor))
        cv2.setMouseCallback(self.window_name, self._handle_mouse)

    def _calculate_scale_factor(self):
        """计算适应屏幕的缩放比例"""
        try:
            monitor = screeninfo.get_monitors()[0]
            max_h = monitor.height - 200  # 留出界面空间
            max_w = monitor.width - 200
            h, w = self.image.shape[:2]
            self.scale_factor = min(max_w / w, max_h / h, 1.0)
        except:
            self.scale_factor = 0.8  # 默认缩放

    def _init_points(self):
        """初始化默认的4个角点"""
        h, w = self.image.shape[:2]
        offset_w, offset_h = w * 0.1, h * 0.1  # 边界偏移量
        self.points = [
            [offset_w, offset_h],  # 左上
            [w - offset_w, offset_h],  # 右上
            [w - offset_w, h - offset_h],  # 右下
            [offset_w, h - offset_h]  # 左下
        ]

    def _handle_mouse(self, event, x, y, flags, param):
        """鼠标事件处理"""
        # 转换坐标到原始图像尺寸
        x_orig = int(x / self.scale_factor)
        y_orig = int(y / self.scale_factor)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击了某个控制点
            for i, (px, py) in enumerate(self.points):
                if np.sqrt((px - x_orig) ** 2 + (py - y_orig) ** 2) < 15 / self.scale_factor:
                    self.selected_point = i
                    break

        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_point = None

        elif event == cv2.EVENT_MOUSEMOVE and self.selected_point is not None:
            # 更新点坐标
            self.points[self.selected_point] = [x_orig, y_orig]
            self._update_display()

    def _update_display(self):
        """更新显示图像"""
        # 缩放图像
        if self.scale_factor != 1.0:
            display_img = cv2.resize(self.image, None,
                                     fx=self.scale_factor,
                                     fy=self.scale_factor)
        else:
            display_img = self.image.copy()

        # 绘制控制点和四边形
        for i, (x, y) in enumerate(self.points):
            x_disp = int(x * self.scale_factor)
            y_disp = int(y * self.scale_factor)
            color = (0, 255, 0) if i == self.selected_point else (0, 200, 0)
            cv2.circle(display_img, (x_disp, y_disp), 10, color, -1)
            cv2.putText(display_img, str(i + 1), (x_disp - 5, y_disp + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if len(self.points) == 4:
            disp_points = np.array([[int(x * self.scale_factor), int(y * self.scale_factor)]
                                    for (x, y) in self.points])
            cv2.polylines(display_img, [disp_points], True, (0, 0, 255), 2)

        cv2.imshow(self.window_name, display_img)

    def get_perspective_crop(self):
        """执行透视变换裁剪"""
        if len(self.points) != 4:
            raise ValueError("需要选择4个顶点")

        # 计算目标尺寸（取四边形对边的最大长度）
        width = int(max(
            np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1])),
            np.linalg.norm(np.array(self.points[2]) - np.array(self.points[3]))
        ))
        height = int(max(
            np.linalg.norm(np.array(self.points[1]) - np.array(self.points[2])),
            np.linalg.norm(np.array(self.points[3]) - np.array(self.points[0]))
        ))

        # 定义目标矩形（保持原始宽高比）
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(
            np.array(self.points, dtype=np.float32),
            dst_points
        )

        # 应用变换
        warped = cv2.warpPerspective(self.image, M, (width, height))
        return warped

    def run(self):
        """运行交互式裁剪程序"""
        print("操作指南：")
        print("1. 拖动绿色控制点调整选区")
        print("2. 按Enter确认裁剪")
        print("3. 按R键重置选区")
        print("4. 按ESC退出")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter键确认
                try:
                    result = self.get_perspective_crop()
                    cv2.imshow("Cropped Result", result)
                    cv2.waitKey(0)
                    return result
                except Exception as e:
                    print(f"错误: {e}")
            elif key == ord("r"):  # 重置
                self._init_points()
                self._update_display()
            elif key == 27:  # ESC退出
                break

        cv2.destroyAllWindows()
        return None