from ultralytics import YOLO

# 加载模型（会自动下载预训练权重到 ~/.ultralytics/ 目录）
model = YOLO('yolov8s.pt')   # 可选 'yolov8m.pt', 'yolov8l.pt'

# 导出为 ONNX，动态轴可提升灵活性
model.export(format='onnx', imgsz=640, dynamic=True)