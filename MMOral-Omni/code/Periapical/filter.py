import os
import json

input_dir = "./Oral-GPT-data/dental_json/MM-Oral-Periapical-jsons-Attributes"
output_dir = "./Oral-GPT-data/dental_json/MM-Oral-Periapical-jsons-filter"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构造保留的结构
    result = {
        "image_id": data.get("image_id"),
        "file_name": data.get("file_name"),
        "image_width": data.get("image_width"),
        "image_height": data.get("image_height"),
        "image_modality": data.get("image_modality"),
        "properties": {
            "Location": {},
            "Classification": {}
        }
    }

    # 保留 Location 下每个病灶的 bbox + score
    location = data.get("properties", {}).get("Location", {})
    for key, val in location.items():
        result["properties"]["Location"][key] = {
            "bbox": val.get("bbox", []),
            "score": val.get("score", [])
        }

    # 保留 Classification 下的所有分类（原样保留）
    classification = data.get("properties", {}).get("Classification", {})
    for key, val in classification.items():
        result["properties"]["Classification"][key] = val

    # 如果 Location 和 Classification 都是空的 → 跳过
    if not result["properties"]["Location"] and not result["properties"]["Classification"]:
        print(f"跳过空文件: {filename}")
        continue

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"处理完成: {filename} → {output_path}")
