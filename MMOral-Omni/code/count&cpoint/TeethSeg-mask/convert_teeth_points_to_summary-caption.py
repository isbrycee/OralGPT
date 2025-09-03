import json
import os
import cv2
from collections import defaultdict

input_json = "./Oral-GPT-data/teeth_points.json"
image_folder = "./Oral-GPT-data/image/mask"
output_json = "./teeth_metadata_transformed.json"


def generate_caption(tooth_count, category_summary):
    caption = f"This is an intraoral photography image containing {tooth_count} teeth. "
    caption += f"There are {len(category_summary)} tooth types identified, including "

    parts = []
    for item in category_summary:
        cat = item['category']
        count = item['count']
        locs = item['locations']
        # 坐标四舍五入取整数，并加上 <points> 标签
        loc_str = ", ".join([f"<points>({round(x)}, {round(y)})</points>" for x, y in locs])
        plural_s = 's' if count > 1 else ''
        parts.append(f"{count} {cat}{plural_s} located at [{loc_str}]")

    caption += "; ".join(parts) + "."
    return caption


with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

grouped = defaultdict(list)
for item in data:
    grouped[item["file_name"]].append(item)

results = []
idx = 0

for file_name, points in grouped.items():
    # 替换 file_name 后缀：_crown.tiff -> _mesh.jpg
    jpg_file_name = file_name.replace("_crown.tiff", "_mesh.jpg")
    image_path = os.path.join(image_folder, jpg_file_name)

    if not os.path.exists(image_path):
        print(f"⚠️ 警告：对应 jpg 图像不存在，跳过 {jpg_file_name}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 警告：无法读取图像文件 {image_path}，跳过")
        continue

    height, width = img.shape[:2]

    category_locations = defaultdict(list)
    for pt in points:
        category = pt["category"]
        x, y = pt["cpoint"]
        category_locations[category].append([float(x), float(y)])

    category_summary = []
    category_count = {}
    for category, locs in category_locations.items():
        category_summary.append({
            "category": category,
            "count": len(locs),
            "locations": locs
        })
        category_count[category] = len(locs)

    tooth_count = sum(category_count.values())
    modality = "Intraoral photography"

    caption = generate_caption(tooth_count, category_summary)

    annotations = []
    for pt in points:
        annotations.append({
            "category": pt["category"],
            "bbox": pt["bbox"]
        })

    result = {
        "image_id": str(idx),
        "file_name": jpg_file_name,
        "width": width,
        "height": height,
        "modality": modality,
        "tooth_count": tooth_count,
        "annotations": annotations,  # 新增
        "category_count": category_count,
        "category_summary": category_summary,
        "caption": caption
    }
    results.append(result)
    idx += 1

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 转换完成，输出保存至：{output_json}")
