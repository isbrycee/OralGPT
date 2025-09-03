import os
import json
import cv2
import numpy as np
from tqdm import tqdm

#从牙齿 mask 彩色图中提取每颗牙齿的中心点 (cpoint)、bbox、类别信息，然后保存成 JSON，同时生成可视化图像。
# 路径设置
input_folder = './Oral-GPT-data/image/mask'  # mask图像输入文件夹
output_json = './teeth_points.json'  # 输出JSON文件
visual_folder = './mask_visualizations'  # 可视化图保存路径
os.makedirs(visual_folder, exist_ok=True)

# 颜色 -> T标签映射
COLOR_TO_T_LABEL = {
    (255, 0, 255): "T0",
    (125, 0, 255): "T16",
    (0, 0, 255): "T7",
    (125, 255, 0): "T15",
    (0, 255, 255): "T6",
    (125, 255, 125): "T14",
    (0, 255, 0): "T5",
    (255, 0, 125): "T13",
    (255, 255, 0): "T4",
    (0, 125, 125): "T12",
    (255, 125, 0): "T3",
    (0, 125, 255): "T11",
    (255, 0, 0): "T2",
    (125, 125, 0): "T10",
    (190, 255, 64): "T1",
    (125, 125, 255): "T9"
}

# T标签 -> 类别
T_LABEL_TO_CATEGORY = {
    "T0": "3rd Molar", "T16": "3rd Molar",
    "T7": "2nd Molar", "T15": "2nd Molar",
    "T6": "1st Molar", "T14": "1st Molar",
    "T5": "2nd Premolar", "T13": "2nd Premolar",
    "T4": "1st Premolar", "T12": "1st Premolar",
    "T3": "Canine", "T11": "Canine",
    "T2": "Lateral Incisor", "T10": "Lateral Incisor",
    "T1": "Central Incisor", "T9": "Central Incisor"
}

# 颜色中文名（可选）
COLOR_NAME_MAP = {
    (255, 0, 255): "粉紫色", (0, 0, 255): "蓝色", (0, 255, 255): "青色", (0, 255, 0): "绿色",
    (255, 255, 0): "黄色", (255, 125, 0): "橙色", (255, 0, 0): "红色", (190, 255, 64): "黄绿色",
    (125, 0, 255): "紫色", (125, 125, 0): "橄榄色", (0, 125, 255): "天蓝色", (0, 125, 125): "深青色",
    (255, 0, 125): "品红色", (125, 255, 125): "浅绿色", (125, 255, 0): "亮绿色", (125, 125, 255): "淡蓝紫色"
}

# 新的特殊图像颜色对应固定bbox数据
SPECIAL_COLOR_BBOXES = {
    "61aa6d7aed0a496d82304456eacae89c__22_1_crown.tiff": {
        (0, 125, 255): [
            [173, 280, 43, 49],
            [199, 306, 42, 47]
        ]
    },
    "63ff2dca51ac40d4b794869474655459__18_1_crown.tiff": {
        (255, 0, 125): [
            [98, 182, 56, 47],
            [118, 224, 61, 47]
        ]
    },
    "94faca5dff394eb3b420ae99ed0fcf8c__24_1_crown.tiff": {
        (125, 255, 0): [
            [102, 92, 67, 57],
            [114, 141, 64, 68]
        ]
    }
}

def process_mask_image(image_path, file_name):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_draw = img.copy()

    unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
    points = []

    for color in unique_colors:
        color_tuple = tuple(color.tolist())
        if color_tuple not in COLOR_TO_T_LABEL:
            continue

        if file_name in SPECIAL_COLOR_BBOXES and color_tuple in SPECIAL_COLOR_BBOXES[file_name]:
            # 用固定bbox处理
            for bbox in SPECIAL_COLOR_BBOXES[file_name][color_tuple]:
                x, y, w, h = bbox
                cx = x + w // 2
                cy = y + h // 2

                points.append({
                    "file_name": file_name,
                    "cpoint": [cx, cy],
                    "bbox": bbox,
                    "color_value": list(color_tuple),
                    "color_name": COLOR_NAME_MAP.get(color_tuple, "Unknown"),
                    "T_label": COLOR_TO_T_LABEL[color_tuple],
                    "category": T_LABEL_TO_CATEGORY.get(COLOR_TO_T_LABEL[color_tuple], "Unknown")
                })

                cv2.circle(img_draw, (cx, cy), 6, (255, 255, 255), -1)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # 使用 cv2.connectedComponentsWithStats 处理每个独立的颜色区域
            mask = np.all(img_rgb == color, axis=-1).astype(np.uint8)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

            max_area = 0
            best_region_id = -1

            # 遍历所有连通区域，找到面积最大的区域
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                # 增加过滤，排除过小或不合理的区域
                if area > max_area and area > 50 and h > 10:
                    max_area = area
                    best_region_id = i

            # 如果找到了面积最大的有效区域
            if best_region_id != -1:
                x = stats[best_region_id, cv2.CC_STAT_LEFT]
                y = stats[best_region_id, cv2.CC_STAT_TOP]
                w = stats[best_region_id, cv2.CC_STAT_WIDTH]
                h = stats[best_region_id, cv2.CC_STAT_HEIGHT]
                cx = int(centroids[best_region_id][0])
                cy = int(centroids[best_region_id][1])

                # 显式地将 numpy 类型转换为 Python 内置 int 类型
                bbox = [int(x), int(y), int(w), int(h)]

                points.append({
                    "file_name": file_name,
                    "cpoint": [cx, cy],
                    "bbox": bbox,
                    "color_value": list(color_tuple),
                    "color_name": COLOR_NAME_MAP.get(color_tuple, "Unknown"),
                    "T_label": COLOR_TO_T_LABEL[color_tuple],
                    "category": T_LABEL_TO_CATEGORY.get(COLOR_TO_T_LABEL[color_tuple], "Unknown")
                })

                cv2.circle(img_draw, (cx, cy), 6, (255, 255, 255), -1)
                cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    save_path = os.path.join(visual_folder, file_name)
    cv2.imwrite(save_path, img_draw)

    return points


def main():
    all_results = []
    mask_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tiff')]

    for file_name in tqdm(mask_files, desc="Processing masks"):
        image_path = os.path.join(input_folder, file_name)
        points = process_mask_image(image_path, file_name)
        all_results.extend(points)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 共处理 {len(mask_files)} 张图像，输出 JSON 至 {output_json}")
    print(f"✅ 可视化图像已保存至：{visual_folder}")

if __name__ == "__main__":
    main()