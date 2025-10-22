import warnings
warnings.filterwarnings('ignore')
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # avoid libomp conflict

from pathlib import Path
# Ensure project root is first on sys.path so local `ultralytics` is imported instead of an installed package
proj_root = Path(__file__).resolve().parent.parent  # repo root
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))
    # debug
    print('inserted proj_root into sys.path:', str(proj_root))
print('sys.path[0:5]=', sys.path[:5])

from ultralytics import YOLO

def get_dataset_info(data_yaml_path):
    """读取数据集的配置信息"""
    import yaml
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    return data_cfg

if __name__ == '__main__':    
    # 数据集配置文件路径
    # data_path = './datasets/spectrum500/data.yaml'
    data_path = './datasets/chiliseed/data.yaml'
    
    # 根据数据集信息选择合适的模型配置
    data_cfg = get_dataset_info(data_path)
    is_multispectral = data_cfg.get('nc2', 0) > 0  # 检查是否是多光谱多分类任务
    
    print(f"Dataset info: nc={data_cfg.get('nc')}, nc2={data_cfg.get('nc2', 0)}")
    print(f"Using {'multispectral' if is_multispectral else 'standard'} configuration")
    
    if is_multispectral:
        cfg_path = './ultralytics/cfg/models/v8/yolov8-msml.yaml'  # 多光谱多分类配置
        model = YOLO(cfg_path)
        print(f"Loading multispectral model from {cfg_path}")
    else:
        # 对于单分类任务，使用yaml配置文件从头训练，确保类别数匹配
        cfg_path = './ultralytics/cfg/models/v8/yolov8.yaml'
        model = YOLO(cfg_path)
        print(f"Loading standard YOLOv8 model from {cfg_path} for {data_cfg.get('nc')} classes")

    # model = YOLO('./runs/train/exp36/weights/best.pt')  # load a pretrained model
    results = model.train(
        data=data_path,
        imgsz=640,
        epochs=3,  # 测试3个epochs
        batch=16,   # 使用和成功的exp相同的batch size
        close_mosaic=20,  # 使用和成功的exp相同的close_mosaic
        workers=8,
        device='cpu',  # Use GPU if available
        optimizer='SGD',  # Using SGD
        
        project='runs/train',
        name='exp',
        # Enable these for more control:
        # resume=False,  # Resume training
        # amp=False,    # Mixed precision training
        # fraction=0.1, # Dataset fraction
    )   
    
    # Get final metrics from the trainer's validator
    final_metrics = None
    if hasattr(model, 'trainer') and hasattr(model.trainer, 'metrics'):
        final_metrics = model.trainer.metrics
    elif results is not None:
        final_metrics = results
    
    # Check if metrics are available
    if final_metrics is not None:
        # Convert metrics to dict if it's a Metrics object
        if hasattr(final_metrics, 'results_dict'):
            metrics_dict = final_metrics.results_dict
        elif isinstance(final_metrics, dict):
            metrics_dict = final_metrics
        else:
            metrics_dict = {}
        
        if metrics_dict:                            
            # Print comprehensive training summary
            print("\n" + "=" * 100)
            print("📊 TRAINING SUMMARY")
            print("=" * 100)
            
            # ⚖️ Fitness Information
            print("\n⚖️  Fitness Information:")
            print("-" * 100)
            if 'fitness/cls1' in metrics_dict and 'fitness/cls2' in metrics_dict:
                # Combined fitness mode
                fitness = metrics_dict.get('fitness', 0.0)
                fitness_cls1 = metrics_dict.get('fitness/cls1', 0.0)
                fitness_cls2 = metrics_dict.get('fitness/cls2', 0.0)
                print(f"  Combined Fitness                             : {fitness:.4f}")
                print(f"  Shape (cls1) Fitness                         : {fitness_cls1:.4f}")
                print(f"  Material (cls2) Fitness                      : {fitness_cls2:.4f}")
            elif 'fitness' in metrics_dict:
                # Single fitness mode (no cls2)
                fitness = metrics_dict.get('fitness', 0.0)
                print(f"  Overall Fitness                              : {fitness:.4f}")
            
            # 📦 Primary Classification Results
            print("\n📦 Primary Classification (Shape) Results:")
            print("-" * 100)
            if 'metrics/precision(B)' in metrics_dict:
                print(f"  Precision                                    : {metrics_dict['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics_dict:
                print(f"  Recall                                       : {metrics_dict['metrics/recall(B)']:.4f}")
            if 'metrics/mAP50(B)' in metrics_dict:
                print(f"  mAP@0.5                                      : {metrics_dict['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics_dict:
                print(f"  mAP@0.5:0.95                                 : {metrics_dict['metrics/mAP50-95(B)']:.4f}")
            
            # 🧪 Secondary Classification Results
            has_material_metrics = any('(M)' in k for k in metrics_dict.keys())
            if has_material_metrics:
                print("\n🧪 Secondary Classification (Material) Results:")
                print("-" * 100)
                if 'metrics/precision(M)' in metrics_dict:
                    print(f"  Precision                                    : {metrics_dict['metrics/precision(M)']:.4f}")
                if 'metrics/recall(M)' in metrics_dict:
                    print(f"  Recall                                       : {metrics_dict['metrics/recall(M)']:.4f}")
                if 'metrics/mAP50(M)' in metrics_dict:
                    print(f"  mAP@0.5                                      : {metrics_dict['metrics/mAP50(M)']:.4f}")
                if 'metrics/mAP50-95(M)' in metrics_dict:
                    print(f"  mAP@0.5:0.95                                 : {metrics_dict['metrics/mAP50-95(M)']:.4f}")
            else:
                print("\n🧪 Secondary Classification (Material) Results:")
                print("-" * 100)
                print("  No material classification metrics available")
            
            print("\n" + "=" * 100)
        else:
            print("  ⚠️  Metrics dictionary is empty")

