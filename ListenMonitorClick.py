import cv2

class ClickMonitor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []
        self.undo_points = []
        self.window_name = 'ClickMonitor'

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            # 左键点击
            self.points.append((x, y))
            print(f'Clicked at ({x}, {y})')
        elif event == cv2.EVENT_RBUTTONUP:
            # 右键点击
            if self.points:
                self.undo_points.append(self.points.pop())
                print(f'Undo last point, points: {len(self.points)}')

    def start(self):
        # 加载图像并缩小到适合屏幕大小
        image = cv2.imread(self.image_path)
        height, width, _ = image.shape
        max_size = 800  # 设定最大大小为800x800
        # if height > max_size or width > max_size:
        #     scale = min(max_size / height, max_size / width)
        #     image = cv2.resize(image, None, fx=scale, fy=scale)
        if height > max_size or width > max_size:
            scale = min(max_size / height, max_size / width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            scale = 1.0

        # 创建窗口并绑定鼠标事件
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

        # 显示图像
        while True:
            # 在图像上绘制已选择的点
            for point in self.points:
                cv2.circle(image, point, 2, (0, 0, 255), -1)

            # 在图像上绘制最后一个已撤销的点
            if self.undo_points:
                cv2.circle(image, self.undo_points[-1], 2, (255, 0, 0), -1)

            # 显示图像
            cv2.imshow(self.window_name, image)

            # 按下 s 键保存并退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 转换坐标点为原始图像中的坐标点
                if scale != 1.0:
                    self.points = [(int(x / scale), int(y / scale)) for x, y in self.points]
                print(f'Selected points: {self.points}')
                cv2.destroyAllWindows()
                return self.points

            # 按下 ESC 键退出
            if key == 27:
                cv2.destroyAllWindows()
                return None

