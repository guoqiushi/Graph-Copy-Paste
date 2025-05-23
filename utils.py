from ultralytics import YOLO
from itertools import combinations
from coco import coco_classes

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

def get_class_index_by_class(target):
    coco_dict = coco_classes
    return next((k for k, v in coco_dict.items() if v == target), None)

import torch
from torch_geometric.data import Data

def build_graph_from_edgelist(edge_list, num_node_features=3, node_labels=None):
    """
    将边列表 [[a,b], [c,d], ...] 转换为 PyG 的 Data 对象

    参数:
    - edge_list: list of [source, target] 的边
    - num_node_features: 每个节点的特征维度（默认为3）
    - node_labels: 可选，列表或张量，表示每个节点的类别（长度等于节点数）

    返回:
    - PyG Data 对象（包含 x, edge_index, y）
    """

    # 构建 edge_index 张量，形状为 [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 推断节点数
    num_nodes = int(edge_index.max().item()) + 1

    # 构建节点特征（随机）
    x = torch.randn((num_nodes, num_node_features))

    # 构建标签（如果未提供则默认为全 0）
    if node_labels is not None:
        y = torch.tensor(node_labels, dtype=torch.long)
    else:
        y = torch.zeros(num_nodes, dtype=torch.long)

    # 构建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)

    return data

def encode_image_feature(img_path,class_name):
    res = detect_objects(img_path)
    if not res.get(class_name):
        return [0,0,0,0,0]
    num_target = len(res.get(class_name))
    x1,y1,x2,y2 = res.get(class_name)[0]
    pos_x = int((x1+x2)*0.5)
    pos_y = int((y1+y2)*0.5)
    num_obj = len(res.keys())
    h,w,_ = cv2.imread(img_path).shape
    ratio = ((y2-y1)*(x2-x1))/(h*w)
    return [num_target,round(ratio,4),pos_x,pos_y,num_obj]
