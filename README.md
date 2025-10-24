<div align="center">
  <h1>MS-YOLO: Multi-Spectral YOLO with Dual Classification</h1>
  <p><b>Forked from Ultralytics YOLO - Enhanced for Multi-Spectral Imaging and Dual Classification</b></p>
  <p>
    <a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=org&utm_campaign=yv25_event" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>

[中文](README.zh-CN.md) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://clickpy.clickhouse.com/dashboard/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Ultralytics YOLO Citation"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo11"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
</div>
</div>
</div>
<br>

## 🌟 MS-YOLO Key Features

This fork extends Ultralytics YOLO with specialized features for multi-spectral imaging and dual classification tasks:

### 🎯 **Dual Classification System**
- **Primary Classification (cls1)**: Object shape classification (e.g., bottle, round can, round cup, square box)
- **Secondary Classification (cls2)**: Material classification (e.g., plastic, metal, paper, glass)
- **Combined Fitness**: Weighted combination of both classification accuracies for holistic model evaluation

### 📊 **Data Augmentation Fixes for Dual Labels**
Complete support for cls2 labels across all augmentation pipelines:
- ✅ **Mosaic Augmentation**: Proper cls2 concatenation and filtering
- ✅ **RandomPerspective**: Synchronized cls2 filtering during affine transformations
- ✅ **CopyPaste**: Correct cls2 handling when copying objects
- ✅ **Format Transform**: cls2 tensor conversion for PyTorch compatibility
- ✅ **Label Verification**: Duplicate removal for cls2 labels

### 🔬 **Multi-Spectral Support**
- Single-channel multi-spectral image processing (e.g., 1-channel .raw files)
- Custom data loading for non-RGB imagery
- Optimized for specialized imaging sensors

### 📈 **Enhanced Training Summary**
- Separate metrics display for primary and secondary classification
- Combined fitness calculation with configurable weights
- Detailed validation results for both classification heads

### 🏗️ **Model Architecture**
- Dual detection heads with independent classification layers
- `nc`: Number of primary classes (shape)
- `nc2`: Number of secondary classes (material)
- Compatible with YOLOv8 architecture

## 🚀 Quick Start with Dual Classification

### Label Format
Use 6-column format: `cls1 cls2 x y w h`
```
# Example: bottle (cls1=0), plastic (cls2=1), bbox coordinates
0 1 0.478474 0.543053 0.506849 0.585127
```

### Training Configuration
```yaml
# data.yaml
path: ./datasets/your_dataset
train: train/images
val: valid/images
test: test/images

# Primary classes (shape)
nc: 4
names:
  0: bottle
  1: round_can
  2: round_cup
  3: square_box

# Secondary classes (material)
nc2: 5
names2:
  0: cr_plastic
  1: metal
  2: paper
  3: glass
  4: w_plastic

# Multi-spectral config
channels: 1  # For single-channel images
```

### Python Training
```python
from ultralytics import YOLO

# Load multi-spectral dual-classification model
model = YOLO("yolov8-msml.yaml")

# Train with dual classification
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 🔧 Bug Fixes in This Fork

### Critical cls2 Label Handling Fixes
1. **Mosaic Augmentation** (`ultralytics/data/augment.py`)
   - Fixed: cls2 labels not concatenated from 4 mosaic images
   - Fixed: cls2 not filtered with `good` mask after clipping

2. **RandomPerspective** (`ultralytics/data/augment.py`)
   - Fixed: cls2 not filtered when boxes are removed due to affine transformations
   - Added: Synchronized cls2 indexing with bbox filtering

3. **CopyPaste** (`ultralytics/data/augment.py`)
   - Fixed: cls2 not concatenated when copying objects
   - Added: cls2 handling in object copy loop

4. **Format Transform** (`ultralytics/data/augment.py`)
   - Fixed: cls2 remained as numpy array while cls was converted to tensor
   - Added: cls2 tensor conversion matching cls behavior

5. **Label Verification** (`ultralytics/data/utils.py`)
   - Fixed: Duplicate labels removed from bboxes but not from cls2
   - Added: Synchronized duplicate removal for cls2

## 📊 Training Output Example

```
====================================================================================================
📊 TRAINING SUMMARY
====================================================================================================

⚖️  Fitness Information:
----------------------------------------------------------------------------------------------------
  Combined Fitness                             : 0.7234
  Shape (cls1) Fitness                         : 0.7891
  Material (cls2) Fitness                      : 0.6577

📦 Primary Classification (Shape) Results:
----------------------------------------------------------------------------------------------------
  Precision                                    : 0.8234
  Recall                                       : 0.7654
  mAP@0.5                                      : 0.8123
  mAP@0.5:0.95                                 : 0.6891

🧪 Secondary Classification (Material) Results:
----------------------------------------------------------------------------------------------------
  Precision                                    : 0.7123
  Recall                                       : 0.6891
  mAP@0.5                                      : 0.7234
  mAP@0.5:0.95                                 : 0.5892
====================================================================================================
