# -*- coding: utf-8 -*-
import re
import sys

import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

sys.path.append("..")


class samImageManager:
    def __init__(self, imgPath, fileName, maskPots, maskPath):
        self._imgPath = imgPath
        self._fileName = fileName
        self._maskPots = maskPots
        self._maskPath = maskPath
        self._new_filename = re.sub(r'\.(jpg|jpeg|png|bmp|gif)$', '', self._fileName)
        self._maskList = []
        self._count = 1
        self._modelPath = "./models/sam_vit_b_01ec64.pth"
        self._modelType = "vit_b"
        self._device = "cuda"

    def _segmentImg(self, x, y):
        # 加载模型
        sam = sam_model_registry[self._modelType](checkpoint=self._modelPath)
        sam.to(device=self._device)
        predictor = SamPredictor(sam)
        # 将输入的图像进行编码
        img = cv2.imread(self._imgPath)
        predictor.set_image(img)
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        # SamPredictor.predict进行分割，模型会返回这些分割目标对应的置信度，返回三个置信度不同的图片
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        maxScoreMask = masks[np.argmax(scores)]
        return maxScoreMask  # 返回mask的二维数组，外部定义一个空数组用于保存全部mask

    def inference(self):
        # 遍历maskPots逐点生成mask,并将多个点位生成的mask融到一张图内
        for pot in self._maskPots:
            print("正在处理第", self._count, "个点位")
            print("当前处理点位：", pot)
            mask = self._segmentImg(pot[0], pot[1])
            self._maskList.append(mask)
            self._count += 1
        self._maskMix()

    def _maskMix(self):
        # 将self.maskList中的mask融合到一张图内
        mask = np.zeros((self._maskList[0].shape[0], self._maskList[0].shape[1]), dtype=np.uint8)
        for tempMask in self._maskList:
            mask = mask + tempMask
        # 形态学开运算去除mask中的噪音
        kernel = np.ones((20, 20), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
        cv2.imwrite(self._maskPath + '/' + self._fileName, opening * 255)
