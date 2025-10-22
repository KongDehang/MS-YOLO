"""Raw image format support for YOLO."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import struct


def detect_image_size(total_pixels: int) -> Optional[Tuple[int, int]]:
    """
    Attempt to detect image dimensions from total pixel count.
    Prioritizes square-like dimensions.
    
    Args:
        total_pixels (int): Total number of pixels.
        
    Returns:
        Optional[Tuple[int, int]]: (width, height) tuple if found, None otherwise.
    """
    sqrt_size = int(round(np.sqrt(total_pixels)))
    if sqrt_size * sqrt_size == total_pixels:
        return sqrt_size, sqrt_size
    
    for i in range(max(1, sqrt_size - 10), sqrt_size + 11):
        if total_pixels % i == 0:
            j = total_pixels // i
            return (i, j) if abs(i - j) < abs(j - i) else (j, i)
    return None


def read_raw(filename: str | Path) -> Optional[np.ndarray]:
    """
    Read a raw image file with auto-detected dimensions.
    Assumes uint8 data type and a 4-byte integer header containing data length.
    
    Args:
        filename (str | Path): Path to the raw image file.
        
    Returns:
        Optional[np.ndarray]: Image array if successful, None otherwise.
    """
    try:
        with open(filename, 'rb') as f:
            # Read 4-byte length header
            char_length = struct.unpack('i', f.read(4))[0]
            
            # Auto-detect dimensions
            dims = detect_image_size(char_length)
            if dims is None:
                return None
            
            width, height = dims
            img_data = f.read(char_length)
            
            if len(img_data) != char_length:
                return None
                
            # Convert to numpy array and reshape
            image = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
            return image[..., None]  # Add channel dimension
            
    except Exception:
        return None