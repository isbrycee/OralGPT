import os
import torch
import numpy as np
from PIL import Image
import json
import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig
from util import box_ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from matplotlib.path import Path

def load_model(config_path, checkpoint_path, device='cuda'):
    """加载DINO模型"""
    # 设置模型配置
    args = SLConfig.fromfile(config_path)
    args.device = device
    
    # 构建模型
    model, _, postprocessors = build_model_main(args)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, postprocessors

def load_category_names(coco_path):
    """加载COCO格式的类别名称"""
    # with open(coco_path) as f:
    #     data = json.load(f)
    #     id2name = {cat['id']: cat['name'] for cat in data['categories']}
    categories =  [
        {
            "id": 1,
            "name": "11",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "21",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "22",
            "supercategory": "none"
        },
        {
            "id": 4,
            "name": "23",
            "supercategory": "none"
        },
        {
            "id": 5,
            "name": "12",
            "supercategory": "none"
        },
        {
            "id": 6,
            "name": "13",
            "supercategory": "none"
        },
        {
            "id": 7,
            "name": "15",
            "supercategory": "none"
        },
        {
            "id": 8,
            "name": "17",
            "supercategory": "none"
        },
        {
            "id": 9,
            "name": "27",
            "supercategory": "none"
        },
        {
            "id": 10,
            "name": "28",
            "supercategory": "none"
        },
        {
            "id": 11,
            "name": "38",
            "supercategory": "none"
        },
        {
            "id": 12,
            "name": "36",
            "supercategory": "none"
        },
        {
            "id": 13,
            "name": "34",
            "supercategory": "none"
        },
        {
            "id": 14,
            "name": "33",
            "supercategory": "none"
        },
        {
            "id": 15,
            "name": "32",
            "supercategory": "none"
        },
        {
            "id": 16,
            "name": "31",
            "supercategory": "none"
        },
        {
            "id": 17,
            "name": "41",
            "supercategory": "none"
        },
        {
            "id": 18,
            "name": "43",
            "supercategory": "none"
        },
        {
            "id": 19,
            "name": "44",
            "supercategory": "none"
        },
        {
            "id": 20,
            "name": "45",
            "supercategory": "none"
        },
        {
            "id": 21,
            "name": "48",
            "supercategory": "none"
        },
        {
            "id": 22,
            "name": "18",
            "supercategory": "none"
        },
        {
            "id": 23,
            "name": "16",
            "supercategory": "none"
        },
        {
            "id": 24,
            "name": "42",
            "supercategory": "none"
        },
        {
            "id": 25,
            "name": "35",
            "supercategory": "none"
        },
        {
            "id": 26,
            "name": "47",
            "supercategory": "none"
        },
        {
            "id": 27,
            "name": "24",
            "supercategory": "none"
        },
        {
            "id": 28,
            "name": "25",
            "supercategory": "none"
        },
        {
            "id": 29,
            "name": "14",
            "supercategory": "none"
        },
        {
            "id": 30,
            "name": "26",
            "supercategory": "none"
        },
        {
            "id": 31,
            "name": "37",
            "supercategory": "none"
        },
        {
            "id": 32,
            "name": "46",
            "supercategory": "none"
        }
    ]
    id2name = {cat['id']: cat['name'] for cat in categories}
    return id2name

def preprocess_image(image_path, transform=None):
    """预处理图像"""
    # 如果未提供转换，使用默认设置
    if transform is None:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # 加载并转换图像
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)
    image_transformed, _ = transform(image, None)
    
    return image, image_transformed, orig_size

def run_inference(model, postprocessors, image_tensor, device='cuda'):
    """运行模型推理"""
    with torch.no_grad():
        # 添加批次维度并移至指定设备
        image_tensor = image_tensor.unsqueeze(0).to(device)
        # 获取模型输出
        outputs = model(image_tensor)
        # 后处理输出
        processed_outputs = postprocessors['bbox'](outputs, torch.tensor([[1.0, 1.0]]).to(device))[0]
    
    return processed_outputs

def visualize_predictions(image_path, predictions, id2name, output_path, confidence_threshold=0.3):
    """可视化检测结果"""
    # 加载原始图像
    orig_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_image.size
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(orig_image)
    ax.axis('off')
    ax.set_title(f"Detection Results: {os.path.basename(image_path)}")
    
    # 获取高于置信度阈值的检测结果
    scores = predictions['scores'].cpu()
    labels = predictions['labels'].cpu()
    boxes = predictions['boxes'].cpu()
    
    # 转换为中心点格式
    boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes)
    
    # 筛选出置信度高于阈值的检测
    mask = scores > confidence_threshold
    filtered_boxes = boxes_cxcywh[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    
    print(f"检测到 {len(filtered_boxes)} 个目标，置信度阈值为 {confidence_threshold}")
    
    # 使用不同颜色表示不同类别
    colors = plt.cm.rainbow(np.linspace(0, 1, len(id2name)))
    class_ids = list(id2name.keys())
    
    # 绘制每个检测结果
    for i, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
        # 获取边界框坐标 (cx, cy, w, h)
        cx, cy, w, h = box.numpy()
        
        # 将坐标转换为原始图像尺寸
        cx_orig = cx * orig_w
        cy_orig = cy * orig_h
        w_orig = w * orig_w
        h_orig = h * orig_h
        
        # 计算左上角坐标
        x_orig = cx_orig - w_orig/2
        y_orig = cy_orig - h_orig/2
        
        # 获取类别索引
        label_id = int(label.item())
        label_name = id2name.get(label_id, f"Unknown ({label_id})")
        
        # 选择颜色
        try:
            color_idx = class_ids.index(label_id) % len(colors)
            color = colors[color_idx]
        except ValueError:
            color = colors[i % len(colors)]  # 使用索引作为后备
        
        # 创建矩形
        rect = patches.Rectangle(
            (x_orig, y_orig), w_orig, h_orig,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签文本
        text = f"{label_name} ({score:.2f})"
        text_x = x_orig
        text_y = y_orig - 5
        
        # 添加文本框以提高可读性
        plt.text(
            text_x, text_y, text,
            bbox=dict(facecolor=color, alpha=0.5),
            fontsize=9, color='white'
        )
        
        # 打印检测信息
        print(f"目标 {i+1}: {label_name}, 置信度: {score:.3f}, 位置: [{cx_orig:.1f}, {cy_orig:.1f}, {w_orig:.1f}, {h_orig:.1f}]")
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存至: {output_path}")
    return True

def _convert_bbox_to_xyxy(box_orig):
        if len(box_orig) != 4:
            return box_orig
        
        if isinstance(box_orig[0], (int, float)) and isinstance(box_orig[2], (int, float)):
            if box_orig[0] < box_orig[2] and box_orig[1] < box_orig[3]:
                return [round(float(box_orig[0])), round(float(box_orig[1])), 
                        round(float(box_orig[2])), round(float(box_orig[3]))]
        
        try:
            cx, cy, w, h = box_orig
            x1 = round(float(cx - w / 2))
            y1 = round(float(cy - h / 2))
            x2 = round(float(cx + w / 2))
            y2 = round(float(cy + h / 2))
            return [x1, y1, x2, y2]
        except:
            return box_orig

def calculate_iou(box1, box2):
    """
    计算两个bbox的IoU。
    box格式为：[x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算每个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算IoU
    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='DINO模型单图像推理与可视化')
    parser.add_argument('--image', type=str, default="Dental_Conditions_Detection_2025_Romania/test/images/v9.jpg", help='输入图像路径')
    #parser.add_argument('--config', type=str, default="config/DINO/1_disease_5scale_swin.py", help='模型配置文件路径')
    #parser.add_argument('--checkpoint', type=str, default="model_weights/Teeth_Visual_Experts_dino_Swinl_x-ray_4diseases.pth", help='模型检查点路径')
    #parser.add_argument('--coco_names', type=str, default="teeth_data/teeth_x-ray_4diseases_numImages705_ins_coco/annotations/instances_train2017.json", help='COCO类别文件路径')
    # parser.add_argument('--config', type=str, default="config/DINO/4_crown_5scale_swin.py", help='模型配置文件路径')
    # parser.add_argument('--checkpoint', type=str, default="model_weights/Teeth_Visual_Experts_dino_Swinl_x-ray_12diseases.pth", help='模型检查点路径')
    # parser.add_argument('--coco_names', type=str, default="teeth_data/teeth_x-ray_12diseases_numImages9206_ins_coco/annotations/instances_train2017.json", help='COCO类别文件路径')
    parser.add_argument('--config', type=str, default="config/DINO/DINO_5scale_swinL_panoramic_x-ray_32ToothID.py", help='模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, default="Teeth_Visual_Experts_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.pth", help='模型检查点路径')
    parser.add_argument('--coco_names', type=str, default="32ToothID_DINO_category.json", help='COCO类别文件路径')
    parser.add_argument('--confidence', type=float, default=0.3, help='检测置信度阈值')
    parser.add_argument('--output', type=str, default="./outputs/final/", help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='推理设备')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 设置图像转换
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"正在加载模型: {args.checkpoint}")
    # 加载模型
    model, postprocessors = load_model(args.config, args.checkpoint, args.device)
    
    # 加载类别名称
    id2name = load_category_names(args.coco_names)
    print(f"加载了 {len(id2name)} 个类别名称")
    
    with open('Dental_Conditions_Detection_2025_Romania_all.json', 'r') as f:
        coco_data = json.load(f)

    for img in coco_data['images']:
        # 处理图像
        file_name = img['file_name']
        image_id = img['id']
        image_path = os.path.join('Dental_Conditions_Detection_2025_Romania/all/images', file_name)

        print(f"处理图像: {image_path}")
        
        # 预处理图像
        _, image_tensor, orig_size = preprocess_image(image_path, transform)
        print(f"原始图像尺寸: {orig_size[0]}x{orig_size[1]}")
        orig_w, orig_h = orig_size
        
        # 运行推理
        outputs = run_inference(model, postprocessors, image_tensor, args.device)
        tooth_id_dict = {}
        tooth_id_list = []

        for i in range(len(outputs['boxes'])):
            box_cxcywh = box_ops.box_xyxy_to_cxcywh(outputs['boxes'][i]).cpu().numpy()

            cx, cy, w, h = box_cxcywh
            box_orig = [
                cx * orig_w,  
                cy * orig_h,  
                w * orig_w,   
                h * orig_h    
            ]
            
            label = outputs['labels'][i].item()
            score = outputs['scores'][i].item()
            
            if score < args.confidence:
                continue
                
            tooth_id = id2name[label] # self.class_maps["teeth_id"].get(label, "unknown")
            
            id_annotation = {
                "tooth_id": tooth_id,
                # "quadrant": self.tooth_ids.get(tooth_id, {}).get("quadrant", 0),
                # "is_wisdom_tooth": self.tooth_ids.get(tooth_id, {}).get("is_wisdom_tooth", False),
                # "side": self.tooth_ids.get(tooth_id, {}).get("side", "unknown"),
                "bbox": np.round(_convert_bbox_to_xyxy(box_orig)),
                # "center": np.round([box_orig[0], box_orig[1]]),
                "score": round(score, 2),
                # "conditions": {}
            }
            tooth_id_list.append(id_annotation)
            # Keep only the highest scoring detection for each tooth_id
            # if tooth_id in tooth_id_dict:
            #     if score > tooth_id_dict[tooth_id]["score"]:
            #         tooth_id_dict[tooth_id] = id_annotation
            # else:
            #     tooth_id_dict[tooth_id] = id_annotation


        # 遍历每个标注
        for annotation in coco_data['annotations']:
            if annotation['image_id'] != image_id:
                continue
            # COCO bbox格式是 [x, y, width, height]，需要转换为 [xmin, ymin, xmax, ymax]
            coco_bbox = annotation['bbox']
            coco_bbox_converted = [
                coco_bbox[0],  # xmin
                coco_bbox[1],  # ymin
                coco_bbox[0] + coco_bbox[2],  # xmax
                coco_bbox[1] + coco_bbox[3],  # ymax
            ]

            # 计算与提供的bbox_list中每个bbox的IoU
            max_iou = 0
            matched_tooth_id = None
            for item in tooth_id_list:
                iou = calculate_iou(coco_bbox_converted, item['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    matched_tooth_id = item['tooth_id']

            # 在annotation中添加tooth_id字段
            annotation['tooth_id'] = matched_tooth_id

            # print(annotation)
            # import pdb; pdb.set_trace()

    with open("Dental_Conditions_Detection_2025_Romania_all_with_ToothID.json", "w") as f:
        json.dump(coco_data, f, indent=4)

    print("COCO JSON 已更新并保存为 updated_coco.json")

    # # 设置输出路径
    # output_filename = f"detection_{os.path.basename(image_path)}"
    # output_path = os.path.join(args.output, output_filename)
    
    # 可视化结果
    # visualize_predictions(image_path, predictions, id2name, output_path, args.confidence)


if __name__ == "__main__":
    main()
