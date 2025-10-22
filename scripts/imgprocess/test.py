import numpy as np
import rawpy
import matplotlib.pyplot as plt
import sys
import os

# 获取当前文件的父目录
current_dir = os.path.dirname(__file__)
# 获取需要导入的库所在的目录（当前目录的上一级）
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

# 将库所在的目录添加到 sys.path
sys.path.insert(0, parent_dir)

from scripts.imgprocess.read_raw_image import read_raw_image as raw_reader
from scripts.imgprocess.demosaic_bands import fast_demosaic as demosaic


if __name__ == "__main__":
    # 图像尺寸
    WIDTH = 1020
    HEIGHT = 1020
    file_path = 'D:/A_Codes/ultralytics-main/ultralytics-main/datasets/spectrum500/test/images/20251013_140439#CAM0RAW.raw'

    # 读取图像
    # image = raw_reader(file_path, WIDTH, HEIGHT)

    from ultralytics.utils.raw_reader import read_raw  # Local import to avoid circular dependencies
    image = read_raw(file_path)
    
    if image is not None:
        print(f"成功读取图像，形状为: {image.shape}")
        print(f"数据类型: {image.dtype}")
        print(f"像素值范围: [{np.min(image)}, {np.max(image)}]")
    
    # 定义与相机硬件一致的3×3波段矩阵
    # band_matrix = [
    #     [717, 697, 733],   # 第0行的3个波段（单位：nm，可省略单位仅保留数值）
    #     [765, 749, 778],   # 第1行的3个波段
    #     [660, 639, 678]    # 第2行的3个波段
    # ]
    # demosaic_img, band_list = demosaic(image, band_matrix)
    # if demosaic_img is not None:
    #     print(f"成功处理图像，形状为: {demosaic_img.shape}")
    #     print(f"数据类型: {demosaic_img.dtype}")
    #     print(f"像素值范围: [{np.min(demosaic_img)}, {np.max(demosaic_img)}]")
    #     print(f"波段列表: {band_list}")