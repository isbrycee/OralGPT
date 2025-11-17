import os
import json
import cv2
import random
import numpy as np
from pathlib import Path


def draw_boxes_on_image(image_path, annotations, output_dir):
    """在图像上绘制orig_box并保存结果"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[警告] 无法读取 {image_path}")
        return

    for region in annotations:
        box = region.get("orig_box", None)
        class_name = region.get("class_name", "unknown")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = box

        # 为每种类别固定生成一种颜色
        random.seed(hash(class_name) % (2**32))
        color = tuple(int(c) for c in np.random.choice(range(80, 255), 3))

        # 绘制半透明框背景
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        alpha = 0.2  # 透明度
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 绘制边框（加粗）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)

        # 绘制类别标签背景
        label = f"{class_name}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        cv2.rectangle(img, (x1, y1 - text_h - baseline), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - baseline // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # 保存结果
    output_path = Path(output_dir) / image_path.name
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"✅ 已保存可视化结果: {output_path}")


def visualize_bounding_boxes(image_dir, json_path, output_dir):
    """主函数：解析JSON并绘制所有图片的框"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_name = item.get("image").replace('/home/liangyuci/Oral-GPT-data/image/Pathology/data-train/', '/home/jinghao/projects/x-ray-VLM/RGB/histopathological_images/trainingset/data-train/')

        image_path = Path(image_dir) / image_name
        regions = item.get("regions", [])
        if not image_path.exists():
            print(f"[跳过] 找不到图像 {image_path}")
            continue
        draw_boxes_on_image(image_path, regions, output_dir)


if __name__ == "__main__":
    # 👇 修改为你的实际路径
    image_folder = r"/home/jinghao/projects/x-ray-VLM/RGB/histopathological_images/trainingset/data-train"
    json_file = r"/home/jinghao/projects/x-ray-VLM/RGB/histopathological_images/trainingset/clean_train_regions_cot_final.json"
    output_folder = r"/home/jinghao/projects/x-ray-VLM/RGB/histopathological_images/trainingset/visual_images"

    visualize_bounding_boxes(image_folder, json_file, output_folder)
