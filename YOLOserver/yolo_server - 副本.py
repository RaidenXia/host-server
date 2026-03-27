import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import json

app = Flask(__name__)

# 加载模型（假设你已转换好 ONNX 模型）
net = cv2.dnn.readNetFromONNX('yolov8s.onnx')  # 替换为你的模型路径
# 尝试启用 CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 类别名称（COCO 80 类）
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

def preprocess(image, target_size=(640, 640)):
    """预处理图像，保持宽高比，letterbox 缩放"""
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    # 归一化
    padded = padded.astype(np.float32) / 255.0
    # 转换为 CHW
    blob = padded.transpose(2, 0, 1)
    return blob, scale, (left, top, right, bottom)

def postprocess_yolov8(outputs, scale, pad, img_shape, conf_thres=0.25, nms_thres=0.45):
    # 打印输出形状以调试
    print("Output shape:", outputs.shape)
    # 根据实际形状调整
    if outputs.shape[1] == 84 and outputs.shape[2] == 8400:
        outputs = outputs[0]  # (84, 8400)
        outputs = outputs.T   # (8400, 84)
    elif outputs.shape[1] == 8400 and outputs.shape[2] == 84:
        outputs = outputs[0]  # (8400, 84)
    else:
        raise ValueError(f"Unexpected output shape: {outputs.shape}")

    boxes = []
    scores = []
    class_ids = []
    for pred in outputs:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        if score < conf_thres:
            continue
        cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
        x1 = (cx - w/2) * 640
        y1 = (cy - h/2) * 640
        x2 = (cx + w/2) * 640
        y2 = (cy + h/2) * 640
        x1 = (x1 - pad[0]) / scale
        y1 = (y1 - pad[1]) / scale
        x2 = (x2 - pad[0]) / scale
        y2 = (y2 - pad[1]) / scale
        x1 = max(0, min(x1, img_shape[1]))
        y1 = max(0, min(y1, img_shape[0]))
        x2 = max(0, min(x2, img_shape[1]))
        y2 = max(0, min(y2, img_shape[0]))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(score)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
    if len(indices) == 0:
        return []
    # 处理不同 OpenCV 版本返回值
    if isinstance(indices, tuple):
        indices = indices[0]
    result = []
    for i in indices.flatten():
        result.append({
            'x': boxes[i][0],
            'y': boxes[i][1],
            'w': boxes[i][2],
            'h': boxes[i][3],
            'label': class_ids[i],
            'name': class_names[class_ids[i]],
            'prob': scores[i]
        })
    return result

@app.route('/detect', methods=['POST'])
def detect():
    # 接收图像（base64 编码）
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # 预处理
    blob, scale, pad = preprocess(img)
    # 推理
    net.setInput(blob)
    outputs = net.forward()
    # 后处理
    detections = postprocess_yolov8(outputs, scale, pad, img.shape[:2])
    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765, threaded=True)