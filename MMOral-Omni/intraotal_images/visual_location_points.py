import os
import json
import cv2
import random
import argparse
from pathlib import Path
import colorsys
import matplotlib.pyplot as plt
import numpy as np


def get_color_palette(categories):
    """
    使用 Matplotlib 的调色板为类别分配美观颜色
    """
    num_categories = len(categories)
    # 从 'tab10' 和 'Set2' 调色板中生成颜色
    cmap_name = 'Paired' if num_categories <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)

    colors = {}
    for i, category in enumerate(categories):
        rgb = np.array(cmap(i % cmap.N)[:3]) * 255
        colors[category] = tuple(int(c) for c in rgb[::-1])  # 转为 OpenCV BGR
    return colors


def visualize_annotations(image_dir, json_path, output_dir):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 为不同类别分配随机颜色

    # 统计所有类别，用于配色
    all_categories = set()
    for item in annotations:
        for cat in item.get("category_summary", []):
            all_categories.add(cat["category"])
    all_categories = sorted(list(all_categories))

    # 为每个类别分配一个漂亮的颜色
    category_colors = get_color_palette(all_categories)


    for item in annotations:
        if "file_name" not in item:
            print("Warning: Missing 'image_name' in JSON item. Skipping this item.")
            continue

        image_name = item["file_name"].split('/')[-1]
        image_path = os.path.join(image_dir, image_name)

        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_name}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image {image_name}")
            continue

        # 处理每个类别的标注
        for cat in item.get("category_summary", []):
            category = cat["category"]
            locations = cat.get("locations", [])

            if category not in category_colors:
                category_colors[category] = generate_color(category)

            color = category_colors[category]

            # 在图像上绘制点
            for loc in locations:
                x, y = int(loc[0]), int(loc[1])
                cv2.circle(img, (x, y), radius=6, color=color, thickness=-1)

            # 在左上角显示类别名
            # cv2.putText(img, category, (20, 40 + 30 * list(category_colors.keys()).index(category)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 输出路径
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image: {output_path}")

    print("✅ Visualization completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize JSON annotations on images")
    parser.add_argument("--image_dir", default="/home/jinghao/projects/x-ray-VLM/RGB/intraoral_image_for_location_counting/images", help="Path to the image folder")
    parser.add_argument("--json_path", default="/home/jinghao/projects/x-ray-VLM/RGB/intraoral_image_for_location_counting/2.1_intraoralImage_location_2datasets_train.json", help="Path to the annotation JSON file")
    parser.add_argument("--output_dir", default="/home/jinghao/projects/x-ray-VLM/RGB/intraoral_image_for_location_counting/visual_images_points", help="Output directory for visualized images")
    args = parser.parse_args()

    visualize_annotations(args.image_dir, args.json_path, args.output_dir)
