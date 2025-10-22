import math
import struct
from typing import Optional

import numpy as np


def detect_image_size(total_pixels: int) -> Optional[tuple[int, int]]:
    """
    根据总像素数尝试计算可能的图像尺寸。 优先寻找接近正方形的尺寸。.

    参数:
        total_pixels: 总像素数

    返回:
        (width, height) 元组，如果找不到合适的尺寸则返回 None
    """
    # 获取total_pixels的平方根并四舍五入
    sqrt_size = int(round(math.sqrt(total_pixels)))

    # 首先检查是否为完美正方形
    if sqrt_size * sqrt_size == total_pixels:
        return (sqrt_size, sqrt_size)

    # 在平方根附近寻找可能的尺寸
    for i in range(max(1, sqrt_size - 10), sqrt_size + 11):
        if total_pixels % i == 0:
            j = total_pixels // i
            # 返回接近正方形的尺寸
            return (i, j) if abs(i - j) < abs(j - i) else (j, i)

    return None


def read_raw_image(file_path: str, image_width: int = None, image_height: int = None) -> Optional[np.ndarray]:
    """
    读取.raw格式的图像.

    参数:
        file_path: 文件路径
        image_width: 图像宽度（可选）
        image_height: 图像高度（可选）

    返回:
        读取到的图像数据，形状为(height, width)的numpy数组
    """
    try:
        # 以二进制模式打开文件
        with open(file_path, "rb") as f:
            # 读取4字节的int值，表示后续char数据的长度
            char_length = struct.unpack("i", f.read(4))[0]
            print(f"读取到的char数据长度: {char_length}")
            if image_width is None and image_height is None:
                (image_width, image_height) = detect_image_size(char_length)

            # 验证数据长度是否与图像尺寸匹配
            expected_length = image_width * image_height
            if char_length != expected_length:
                print(f"警告: 数据长度({char_length})与预期尺寸({expected_length})不匹配")

            # 根据读取到的长度读取char数据
            img_data = f.read(char_length)

            # 检查是否读取了完整的数据
            if len(img_data) != char_length:
                raise OSError(f"无法读取完整数据，只读取了{len(img_data)}字节，预期{char_length}字节")

            # 将数据转换为numpy数组并重塑为图像尺寸
            image = np.frombuffer(img_data, dtype=np.uint8).reshape((image_height, image_width))

            return image

    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    # 测试文件路径
    file_path = (
        "D:/A_Codes/ultralytics-main/ultralytics-main/datasets/spectrum500/test/images/20251013_140439#CAM0RAW.raw"
    )

    # 测试自动检测尺寸
    print("测试自动检测尺寸:")
    image = read_raw_image(file_path)

    # 测试指定尺寸
    # print("\n测试指定尺寸:")
    # image = read_raw_image(file_path, 1020, 1020)

    if image is not None:
        print(f"成功读取图像，形状为: {image.shape}")
        print(f"数据类型: {image.dtype}")
        print(f"像素值范围: [{np.min(image)}, {np.max(image)}]")
