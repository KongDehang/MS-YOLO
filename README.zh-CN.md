<div align="center">
  <h1>MS-YOLO: 多光谱 YOLO 双分类系统</h1>
  <p><b>基于 Ultralytics YOLO - 增强多光谱成像和双分类功能</b></p>
  <p>
    <a href="https://www.ultralytics.com/events/yolovision?utm_source=github&utm_medium=org&utm_campaign=yv25_event" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>

[English](README.md) | [中文](README.zh-CN.md) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

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

## 🌟 MS-YOLO 核心特性

本项目在 Ultralytics YOLO 基础上扩展了多光谱成像和双分类任务的专用功能：

### 🎯 **双分类系统**
- **主分类 (cls1)**：物体形状分类（例如：瓶子、圆形罐、圆形杯、方形盒）
- **副分类 (cls2)**：材质分类（例如：塑料、金属、纸张、玻璃）
- **组合适应度**：两个分类精度的加权组合，用于整体模型评估

### 📊 **双标签数据增强修复**
完整支持所有数据增强管道中的 cls2 标签：
- ✅ **Mosaic 增强**：正确的 cls2 拼接和过滤
- ✅ **RandomPerspective**：仿射变换时同步 cls2 过滤
- ✅ **CopyPaste**：复制对象时正确处理 cls2
- ✅ **Format 转换**：cls2 张量转换以兼容 PyTorch
- ✅ **标签验证**：cls2 标签的重复移除

### 🔬 **多光谱支持**
- 单通道多光谱图像处理（例如：1 通道 .raw 文件）
- 非 RGB 图像的自定义数据加载
- 针对专用成像传感器优化

### 📈 **增强训练摘要**
- 主分类和副分类的独立指标显示
- 可配置权重的组合适应度计算
- 两个分类头的详细验证结果

### 🏗️ **模型架构**
- 双检测头与独立分类层
- `nc`：主类别数量（形状）
- `nc2`：副类别数量（材质）
- 兼容 YOLOv8 架构

## 🚀 双分类快速开始

### 标签格式
使用 6 列格式：`cls1 cls2 x y w h`
```
# 示例：瓶子 (cls1=0)，塑料 (cls2=1)，边界框坐标
0 1 0.478474 0.543053 0.506849 0.585127
```

### 训练配置
```yaml
# data.yaml
path: ./datasets/your_dataset
train: train/images
val: valid/images
test: test/images

# 主类别（形状）
nc: 4
names:
  0: bottle
  1: round_can
  2: round_cup
  3: square_box

# 副类别（材质）
nc2: 5
names2:
  0: cr_plastic
  1: metal
  2: paper
  3: glass
  4: w_plastic

# 多光谱配置
channels: 1  # 单通道图像
```

### Python 训练
```python
from ultralytics import YOLO

# 加载多光谱双分类模型
model = YOLO("yolov8-msml.yaml")

# 使用双分类进行训练
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 🔧 本项目的 Bug 修复

### 关键 cls2 标签处理修复
1. **Mosaic 增强** (`ultralytics/data/augment.py`)
   - 修复：cls2 标签未从 4 张 mosaic 图像拼接
   - 修复：裁剪后 cls2 未使用 `good` mask 过滤

2. **RandomPerspective** (`ultralytics/data/augment.py`)
   - 修复：仿射变换导致边界框移除时，cls2 未被过滤
   - 添加：cls2 与 bbox 过滤的同步索引

3. **CopyPaste** (`ultralytics/data/augment.py`)
   - 修复：复制对象时 cls2 未拼接
   - 添加：对象复制循环中的 cls2 处理

4. **Format 转换** (`ultralytics/data/augment.py`)
   - 修复：cls2 保持为 numpy 数组，而 cls 已转换为张量
   - 添加：cls2 张量转换，与 cls 行为一致

5. **标签验证** (`ultralytics/data/utils.py`)
   - 修复：重复标签从 bboxes 中移除，但 cls2 中未移除
   - 添加：cls2 的同步重复移除

## 📊 训练输出示例

```
====================================================================================================
📊 训练摘要
====================================================================================================

⚖️  适应度信息:
----------------------------------------------------------------------------------------------------
  组合适应度                                   : 0.7234
  形状 (cls1) 适应度                          : 0.7891
  材质 (cls2) 适应度                          : 0.6577

📦 主分类（形状）结果:
----------------------------------------------------------------------------------------------------
  精确度                                       : 0.8234
  召回率                                       : 0.7654
  mAP@0.5                                      : 0.8123
  mAP@0.5:0.95                                 : 0.6891

🧪 副分类（材质）结果:
----------------------------------------------------------------------------------------------------
  精确度                                       : 0.7123
  召回率                                       : 0.6891
  mAP@0.5                                      : 0.7234
  mAP@0.5:0.95                                 : 0.5892
====================================================================================================