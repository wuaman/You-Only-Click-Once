import os
import tkinter as tk
from tkinter import messagebox
import cv2
from ListenMonitorClick import ClickMonitor
from Mask2Yolo import Mask2Yolo
from segImageProcess import samImageManager


def if_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_popup(message):
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()
    # 显示弹窗
    messagebox.showinfo("HLWY数据标注工具", message)
    # 关闭主窗口
    root.destroy()


def makeLabel(imgFilepath, labelFilepath, maskFilepath, label, indexLabel):
    # 遍历文件夹中的所有图像文件并调用Mask2Yolo对象的convert方法
    if_exist(maskFilepath + label)
    for fileName in os.listdir(imgFilepath):
        if fileName.endswith('.png') or fileName.endswith('.jpg'):
            imgPath = os.path.join(imgFilepath, fileName)
            monitor = ClickMonitor(imgPath)
            points = monitor.start()
            samImg = samImageManager(imgPath, fileName, points, maskFilepath + label)
            samImg.inference()
            maskName = fileName
            maskPath = os.path.join(maskFilepath, label, fileName)
            mask2yolo = Mask2Yolo(maskPath, maskName, labelFilepath, indexLabel)
            mask2yolo.convert()


if __name__ == '__main__':
    # 定义文件夹路径、标签名称和类别数量
    img_path, mask_path, label_path = './DataSet/Image/', './DataSet/mask/', './DataSet/label/'
    labels = ['Clip', 'banzi']
    if_exist(mask_path)
    if_exist(label_path)
    num_classes = len(labels)

    show_popup(f'欢迎使用HLWY图像标注工具，当前标注{labels[0]}类')
    # 遍历文件夹中的所有图像文件并调用Mask2Yolo对象的convert方法
    for i in range(len(labels)):
        makeLabel(img_path, label_path, mask_path, labels[i], i)
        if i < len(labels) - 1:
            show_popup(f'{labels[i]}类标签处理完毕， 点击确定以处理{labels[i + 1]}类')
        else:
            show_popup(f'处理完毕')
