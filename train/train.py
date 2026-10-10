"""Дообучение YOLOv8 на своём датасете загрузки грузовика.

Colab/GPU:
    pip install ultralytics
    python train/train.py
Результат: runs/detect/train/weights/best.pt -> указать в MODEL_PATH для инференса.
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")          # стартуем с предобученных весов (COCO)
    model.train(
        data="train/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        patience=20,
        name="load_safety",
    )
    # метрики на валидации печатаются автоматически (mAP50, P, R)
    metrics = model.val()
    print(metrics)
