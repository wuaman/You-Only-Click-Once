import os
import re

import cv2


class Mask2Yolo:
    def __init__(self, maskPath,maskName, labelPath, labelIndex):
        self.maskPath = maskPath
        self.maskName = maskName
        self.labelPath = labelPath
        self.labelIndex = labelIndex

    def convert(self):
        # 加载图像
        img = cv2.imread(self.maskPath)

        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 转换为二值图像
        _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 查找掩码图像中的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 提取边界框坐标
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, w, h])

        # 定义YOLO格式的标注数据
        annotations = []
        for bbox in bounding_boxes:
            # 计算边界框中心坐标和宽高
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            width = bbox[2]
            height = bbox[3]

            # 计算边界框中心坐标和宽高相对于图像宽高的比例
            x_center /= img.shape[1]
            y_center /= img.shape[0]
            width /= img.shape[1]
            height /= img.shape[0]

            # 将YOLO格式的标注数据添加到列表中
            annotations.append([self.labelIndex, x_center, y_center, width, height])

        # 将YOLO格式的标注数据保存到文件中
        fileName = re.sub(r'\.(jpg|jpeg|png|bmp|gif)$', '', self.maskName)
        annotation_path = os.path.join(self.labelPath, fileName + '.txt')
        print(annotation_path)
        with open(annotation_path, 'a') as f:
            for annotation in annotations:
                f.write(' '.join([str(x) for x in annotation]) + '\n')

        # 打印处理结果
        print(f'Converted {self.maskPath} to {annotation_path}')


