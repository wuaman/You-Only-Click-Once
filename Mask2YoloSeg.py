import os
import re

import cv2
import numpy as np


class Mask2YoloSeg:
    def __init__(self, maskPath, maskName, labelPath, labelIndex):
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
        annotations = []
        for contour in contours:
            # 计算要删除的点的数量
            n_to_delete = (contour.shape[0]-30) if contour.shape[0] > 10 else int(0.9 * contour.shape[0])
            # 使用 numpy.random.choice() 函数随机选择要删除的点的索引
            indices_to_delete = np.random.choice(contour.shape[0], size=n_to_delete, replace=False)
            # 删除选定的点
            new_c = np.delete(contour, indices_to_delete, axis=0)
            # 构造类别+点位
            new_c_norm = new_c / [img.shape[1], img.shape[0]]
            new_c_norm = new_c_norm.astype(str)
            new_c_norm = np.insert(new_c_norm, 0, self.labelIndex)
            annotations.append(new_c_norm)

        # 将YOLO格式的标注数据保存到文件中
        fileName = re.sub(r'\.(jpg|jpeg|png|bmp|gif)$', '', self.maskName)
        annotation_path = os.path.join(self.labelPath, fileName + '.txt')
        with open(annotation_path, 'a') as f:
            for annotation in annotations:
                f.write(' '.join([str(x) for x in annotation]) + '\n')

        # 打印处理结果
        print(f'Converted {self.maskPath} to {annotation_path}')
