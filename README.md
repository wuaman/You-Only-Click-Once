# You-Only-Click-Once

基于 Segment Anything Model 的半自动数据标注工具，只需点击一次即可实现图像分割和标注以及yolo数据格式转换。 支持多目标多类别半自动标注。

## 1. 安装依赖



1. ```python
   pip install -r requirements.txt
   ```

2. 下载SAM模型,放在 **./models/** 路径下，下载地址：这是一个链接 [基础研究部网盘](**aidt.top:8100**)。



## 2. 使用方法

- 根据自身情况填写**main.py** 中 **img_path, mask_path, label_path** 的路径
- 修改标签类别
- 运行即可，鼠标左键选择点位，右键撤回上次所选点位，按 **s** 键保存，按**Ese**键退出
- 请按照 **labels** 中的类别进行标注。即第一轮标注第1类目标，第N轮标注第N类目标，直至标完为止。每一类目标标注完成后会有弹窗提示。

**注意事项：**

1. 如果分割完之后有小噪音被误分割（DataSet/mask中可以查看分割图），可以调整segImageProcess.py中_maskMix方法中形态学操作部分的**kernel_size**和**iterations**，然后重新运行即可。 由于我的像素分辨率较高，内核大小可能偏大，请根据自身情况调节 确保label文件夹内没有与图片同名的txt文件 。
2. 每次运行完请及时清理文件夹，防止下次运行时出现错误。