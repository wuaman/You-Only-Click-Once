import os
import tkinter as tk
from tkinter import messagebox
import cv2
from ListenMonitorClick import ClickMonitor
from Mask2Yolo import Mask2Yolo
from Mask2YoloSeg import Mask2YoloSeg
from segImageProcess import samImageManager


def if_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_popup(message):
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()
    # 显示弹窗
    messagebox.showinfo("Aman's 数据标注工具", message)
    # 关闭主窗口
    root.destroy()

def show_choose_popup():
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()
    # 显示自定义消息内容和标题的对话框
    result = messagebox.askquestion("欢迎使用HLWY数据标注工具", "请选择要制作的数据集类型：\n\n1. 制作检测数据集\n2. 制作分割数据集\n选择“是”用以制作检测类型数据集")
    if result == 'yes':
        print("用户选择了“制作检测数据集”")
        return True
    else:
        print("用户选择了“制作分割数据集”")
        return False


def makeLabel(imgFilepath, labelFilepath, maskFilepath, label, indexLabel, is_seg):
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
            if not is_seg:
                print("选择了seg")
                mask2yolo = Mask2YoloSeg(maskPath, maskName, labelFilepath, indexLabel)
                mask2yolo.convert()
            else:
                print("no seg")
                mask2yolo = Mask2Yolo(maskPath, maskName, labelFilepath, indexLabel)
                mask2yolo.convert()


if __name__ == '__main__':
    # 定义文件夹路径、标签名称和类别数量
    img_path, mask_path, label_path = './DataSet/Image/', './DataSet/mask/', './DataSet/label/'
    labels = ['Clip']
    if_exist(mask_path)
    if_exist(label_path)
    num_classes = len(labels)
    is_seg = show_choose_popup()
    show_popup(f'欢迎使用HLWY图像标注工具，当前标注{labels[0]}类')
    # 遍历文件夹中的所有图像文件并调用Mask2Yolo对象的convert方法
    for i in range(len(labels)):
        makeLabel(img_path, label_path, mask_path, labels[i], i,is_seg)
        if i < len(labels) - 1:
            show_popup(f'{labels[i]}类标签处理完毕， 点击确定以处理{labels[i + 1]}类')
        else:
            show_popup(f'处理完毕')
