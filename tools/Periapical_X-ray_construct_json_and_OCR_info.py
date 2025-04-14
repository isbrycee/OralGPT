import os
import json
from PIL import Image
from datetime import datetime

# 输入图片文件夹路径和输出 JSON 文件夹路径
input_image_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-images"  # 替换为实际图片文件夹路径
output_json_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-jsons"  # 替换为实际 JSON 文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_json_folder):
    os.makedirs(output_json_folder)


# 遍历图片文件夹中的所有图片
for file_name in os.listdir(input_image_folder):
    # 检测文件是否是图片文件
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 获取图片路径
        image_path = os.path.join(input_image_folder, file_name)

        # 打开图片以获取宽高
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        # 调用 OCR 模型获取采集时间
        image_id = int(os.path.splitext(file_name)[0])
        # 构造 JSON 注释内容
        annotation = {
            "image_id": image_id,
            "file_name": file_name,
            "image_width": image_width,
            "image_height": image_height,
            "image_modality": "Periapical X-ray",
            "properties": {
                "Location": [],
                "Classification": [],
            },
            "loc_caption": "......",
            "med_report": "....."
        }

        # 保存 JSON 文件到目标文件夹
        json_file_name = f"{str(image_id).zfill(6)}.json"
        json_file_path = os.path.join(output_json_folder, json_file_name)
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(annotation, json_file, ensure_ascii=False, indent=4)

        print(f"Generated JSON for image: {file_name}")

print("All JSON files have been generated.")
