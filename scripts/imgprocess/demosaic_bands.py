import numpy as np
import cv2
from scipy.interpolate import griddata

def demosaic_specific_bands(raw_img, band_matrix):
    """
    根据具体3×3波段矩阵恢复多光谱图像
    参数:
        raw_img: 原始单通道raw图像 (1020, 1020)
        band_matrix: 3×3波段矩阵（如上述band_matrix）
    返回:
        多光谱图像 (1020, 1020, 9)，9个通道对应9个波段
    """
    h, w = raw_img.shape
    n_bands = 9
    spectral_img = np.zeros((h, w, n_bands), dtype=np.float32)
    band_list = np.unique(band_matrix)  # 提取9个唯一波段（按波长排序）
    
    # 为每个波段构建原始采样点坐标
    for band_idx, wavelength in enumerate(band_list):
        # 找到该波段在3×3矩阵中的位置（相对偏移）
        grid_positions = np.argwhere(np.array(band_matrix) == wavelength)  # 形如[[i,j], [i,j], ...]
        
        # 收集原始图像中属于该波段的所有像素坐标和值
        coords = []
        values = []
        for i in range(h):
            for j in range(w):
                # 计算当前像素在3×3网格中的相对位置
                grid_i, grid_j = i % 3, j % 3
                if band_matrix[grid_i][grid_j] == wavelength:
                    coords.append([i, j])
                    values.append(raw_img[i, j])
        
        # 对缺失像素进行插值（双三次插值更精准）
        coords = np.array(coords)
        values = np.array(values)
        # 生成网格坐标
        grid_x, grid_y = np.mgrid[0:h, 0:w]
        # 插值填充整个波段图像
        band_img = griddata(
            coords, values, (grid_x, grid_y), 
            method='cubic',  # 比双线性更平滑，保留细节
            fill_value=np.mean(values)  # 边缘用均值填充
        )
        spectral_img[..., band_idx] = band_img.astype(np.uint8)
    
    return spectral_img, band_list  # 返回多光谱图像和波段列表


def fast_demosaic(raw_img, band_matrix):
    """
    快速去马赛克函数，处理1020×1020图像，耗时≤10ms
    """
    h, w = raw_img.shape
    # 提取所有唯一波段（确保9个）
    bands = list({b for row in band_matrix for b in row})
    bands.sort()  # 按波长排序（可选，保证通道顺序一致）
    n_bands = len(bands)
    spectral_img = np.zeros((h, w, n_bands), dtype=np.uint8)
    
    # 预计算每个波段在3×3网格中的位置（di, dj）
    band_pos = {b: [] for b in bands}
    for di in range(3):
        for dj in range(3):
            b = band_matrix[di][dj]
            band_pos[b].append((di, dj))
    
    # 对每个波段处理
    for band_idx, b in enumerate(bands):
        # 1. 创建该波段的有效像素掩码（原始尺寸）
        band_img = np.zeros((h, w), dtype=np.uint8)
        for di, dj in band_pos[b]:
            # 关键修正：用切片直接赋值（形状完全匹配）
            # di::3 表示从di开始，每3个像素取一个（行）
            # dj::3 表示从dj开始，每3个像素取一个（列）
            # 原始图像中该区域形状为 (340, 340)，与掩码切片形状一致
            band_img[di::3, dj::3] = raw_img[di::3, dj::3]
        
        # 2. 快速插值：利用OpenCV的缩放实现（替代复杂索引）
        # 步骤1：下采样到1/3尺寸（保留有效像素结构）
        small_h, small_w = h // 3, w // 3  # 1020//3=340，无余数
        small_img = cv2.resize(band_img, (small_w, small_h), 
                             interpolation=cv2.INTER_NEAREST)  #  nearest确保有效像素不变
        
        # 步骤2：上采样回原始尺寸（用线性插值填充空白）
        band_img = cv2.resize(small_img, (w, h), 
                            interpolation=cv2.INTER_LINEAR)  # 线性插值恢复细节
        
        # 3. 存入多光谱图像
        spectral_img[..., band_idx] = band_img.astype(np.uint8)
    
    return spectral_img, bands
