import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

# Ensure project root is first on sys.path so local `ultralytics` is imported instead of an installed package
proj_root = Path(__file__).resolve().parent.parent  # repo root
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))
    # debug
    print("inserted proj_root into sys.path:", str(proj_root))
print("sys.path[0:5]=", sys.path[:5])

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./runs/train/exp38/weights/best.pt")  # select your model.pt path
    model.predict(
        source="./datasets/spectrum500/test/images/",
        imgsz=640,
        project="./runs/detect",
        name="exp",
        save=True,
        # conf=0.2,
        # visualize=True # visualize model features maps
    )
