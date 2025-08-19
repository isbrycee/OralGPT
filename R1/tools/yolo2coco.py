import os
import json
from PIL import Image

# 类别体系
segment_names_to_labels = [
    ("Implant", 0), 
    ("Prosthetic restoration", 1), 
    ("Obturation", 2), 
    ("Endodontic treatment", 3), 
    ("Orthodontic device", 12), 
    ("Surgical device", 13),  # Low risk
    ("Carious lesion", 4), 
    ("Bone resorbtion", 5), 
    ("Impacted tooth", 6), 
    ("Apical surgery", 10),  # Medium risk
    ("Apical periodontitis", 7), 
    ("Root fragment", 8), 
    ("Furcation lesion", 9), 
    ("Root resorption", 11)  # High risk
]

# 类别映射
id_to_name = {label_id: name for name, label_id in segment_names_to_labels}

def yolo_to_coco(image_dir, label_dir, output_json):
    # 初始化 COCO 格式数据结构
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别信息
    for label_id, name in id_to_name.items():
        coco_format["categories"].append({
            "id": label_id,
            "name": name,
            "supercategory": "object"
        })
    
    # 遍历图片文件夹
    annotation_id = 1
    for image_id, image_file in enumerate(sorted(os.listdir(image_dir))):
        if not image_file.endswith((".jpg", ".png", ".jpeg")):
            continue
        
        # 获取图片路径和尺寸
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path)
        width, height = img.size
        
        # 添加图片信息
        coco_format["images"].append({
            "id": image_id + 1,  # 从1开始
            "file_name": image_file,
            "width": width,
            "height": height
        })
        
        # 对应的标签文件
        label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        if not os.path.exists(label_file):
            print(f"Warning: Label file for {image_file} not found. Skipping.")
            continue
        
        # 读取 YOLO 标签
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                # YOLO 格式数据：class_id, x_center, y_center, width, height
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                class_id = int(class_id)
                
                # 转换为 COCO 格式的 bbox: [x_min, y_min, width, height]
                x_min = (x_center - box_width / 2) * width
                y_min = (y_center - box_height / 2) * height
                box_width *= width
                box_height *= height
                
                # 添加注释信息
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id + 1,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    # 保存为 JSON 文件
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)
    print(f"COCO format JSON saved to {output_json}")


# 输入文件夹路径
image_folder = "/home/jinghao/projects/x-ray-VLM/R1/Dental_Conditions_Detection_2025_Romania/all/images"
label_folder = "/home/jinghao/projects/x-ray-VLM/R1/Dental_Conditions_Detection_2025_Romania/all/labels"
output_json_file = "Dental_Conditions_Detection_2025_Romania_all.json"

# 执行转换
yolo_to_coco(image_folder, label_folder, output_json_file)