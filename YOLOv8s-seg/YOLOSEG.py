from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path("/content/drive/MyDrive/dataset")

data_yaml = BASE_DIR / "data.yaml"
with open(data_yaml, "w") as f:
    f.write(f"""
path: {BASE_DIR}
train: images/train
val: images/val
nc: 1
names: ['class_name']
""")

print("🚀 Starting training...")
model = YOLO("yolov8s-seg.pt")
model.train(
    data=str(data_yaml),
    epochs=500,  
    patience=50, 
    imgsz=640,
    batch=4,     
    optimizer="Adam",
    lr0=0.0005,  
    lrf=0.00005, 
    warmup_epochs=5, 
    warmup_bias_lr=0.05,
    weight_decay=0.0005,
    label_smoothing=0.1,
    degrees=20,      
    shear=0.2,       
    translate=0.2,   
    perspective=0.0008, 
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,      
    mixup=0.2,       
    copy_paste=0.2,  
    hsv_h=0.03,      
    hsv_s=0.8,       
    hsv_v=0.6,       
    project="runs/train",
    name="yolov8s-seg-train",
    verbose=True
)

print("Training completed.")
