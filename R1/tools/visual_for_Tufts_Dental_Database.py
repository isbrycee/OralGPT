import json
import cv2
import os
from pathlib import Path

def visualize_boxes(json_path, image_base_dir, output_dir):
    """
    可视化JSON中的定位框并保存图像
    :param json_path: JSON文件路径
    :param image_base_dir: 图像基础目录
    :param output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 遍历每个图像条目
    for item in data:
        # 构建图像路径
        img_path = os.path.join(image_base_dir, item["image_name"])
        
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            continue
            
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 复制原始图像用于绘制
        img_with_boxes = img.copy()
        height, width = img.shape[:2]
        
        # 验证坐标是否在图像范围内
        def validate_coords(coords):
            x1, y1, x2, y2 = coords
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            return [x1, y1, x2, y2]
        
        if item["Contextual Grounding Position"] is None or item["Precise Grounding Position"] is None:
            continue

        # 绘制Contextual Grounding Position (绿色)
        for box in item["Contextual Grounding Position"]:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = validate_coords(box)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形
        
        # 绘制Precise Grounding Position (红色)
        for box in item["Precise Grounding Position"]:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = validate_coords(box)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色矩形
        
        # 添加图例
        legend_text = "Green: Contextual, Red: Precise"
        cv2.putText(img_with_boxes, legend_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, os.path.basename(item["image_name"]))
        cv2.imwrite(output_path, img_with_boxes)
        print(f"已保存可视化图像: {output_path}")

if __name__ == "__main__":
    # 配置参数
    json_path = "processed_annotations.json"  # 替换为你的JSON文件路径
    image_base_dir = "/home/jinghao/projects/x-ray-VLM/R1"  # 图像基础目录
    output_dir = "/home/jinghao/projects/x-ray-VLM/OralGPT/R1/tools/temp"  # 输出目录
    
    # 执行可视化
    visualize_boxes(json_path, image_base_dir, output_dir)
