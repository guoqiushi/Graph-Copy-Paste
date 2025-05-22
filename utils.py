from ultralytics import YOLO
from itertools import combinations


def detect_objects(image_path, model_path="yolov8n.pt"):
    model = YOLO(model_path)
    results = model(image_path)

    output = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])  # 类别ID
            xyxy = box.xyxy[0].tolist()  # 边框坐标 [x1, y1, x2, y2]
            class_name = model.names[cls_id]  # 类别名称

            if class_name not in output:
                output[class_name] = []
            output[class_name].append([int(x) for x in xyxy])

    return output


def get_all_class_name(result):
    result = []
    for class_name,_ in result.items():
        result = list(set(result.append(class_name)))
    return result

def class_to_edge(yolo_res):
    return list(combinations(yolo_res))