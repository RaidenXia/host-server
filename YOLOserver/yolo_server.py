import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
from ultralytics import YOLO

app = Flask(__name__)

# 加载 YOLOv8 模型（首次运行会自动下载 yolov8s.pt 到 ~/.ultralytics/）
model = YOLO('yolov8s.pt')   # 可换成 'yolov8m.pt', 'yolov8l.pt' 等

# COCO 类别名称
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    img_data = base64.b64decode(data['image'])
    # 假设图像是 320x240 的 RGB888（3字节每像素）
    try:
        # 将字节流转换为 numpy 数组，再 reshape 为 (240, 320, 3)
        img = np.frombuffer(img_data, np.uint8).reshape((240, 320, 3))
    except Exception as e:
        return jsonify({'error': f'Invalid image shape: {str(e)}'}), 400

    # 推理
    results = model(img, verbose=False)   # results 是一个列表，包含一张图的检测结果
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()   # 绝对坐标 [x1, y1, x2, y2]
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                # print(f"Box: x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}")
                detections.append({
                    'x': float(x1),
                    'y': float(y1),
                    'w': float(x2 - x1),
                    'h': float(y2 - y1),
                    'label': int(cls[i]),
                    'name': class_names[cls[i]],
                    'prob': float(conf[i])
                })
    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765, threaded=True)