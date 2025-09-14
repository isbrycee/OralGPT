import os
import json
import glob

# 目标目录
folder = ".Oral-GPT-data/dental_json/MM-Oral-Periapical-jsons-filter/"

# 遍历所有 json 文件
for json_file in glob.glob(os.path.join(folder, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False

    # 检查 bbox 并处理
    try:
        location = data["properties"].get("Location", {})
        for key, value in location.items():
            if "bbox" in value:
                new_bbox = []
                for box in value["bbox"]:
                    # 对每个坐标四舍五入取整
                    rounded_box = [int(round(v)) for v in box]
                    new_bbox.append(rounded_box)
                value["bbox"] = new_bbox
                modified = True
    except Exception as e:
        print(f"处理 {json_file} 时出错: {e}")

    # 如果有修改，覆盖保存
    if modified:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ 所有 JSON 文件处理完成！")
