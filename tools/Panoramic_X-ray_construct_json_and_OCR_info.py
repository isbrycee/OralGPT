import os
import json
from PIL import Image
from datetime import datetime
from openocr import OpenOCR
import ast
import re


# 输入图片文件夹路径和输出 JSON 文件夹路径
input_image_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-OPG-images-4k"  # 替换为实际图片文件夹路径
output_json_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-OPG-jsons-4k"  # 替换为实际 JSON 文件夹路径


engine = OpenOCR()

# 确保输出文件夹存在
if not os.path.exists(output_json_folder):
    os.makedirs(output_json_folder)

# 模拟 OCR 获取采集时间的函数
# 实际情况需要调用 OCR 模型，这里假设返回一个固定时间
def get_aquisition_time(image_path):
    # 替换为实际 OCR 检测逻辑
    result, elapse = engine(image_path)
    returned_value = "N/A"
    if result:
    # 010372.jpg	[{"transcription": "7", "points": [[385, 370], [405, 349], [415, 359], [395, 380]], "score": 0.9643768668174744}]
        for item in result:
            all_ocr_dict = item.split('\t')[1]
            all_ocr_list = ast.literal_eval(all_ocr_dict)

            for item_dict in all_ocr_list:
                ocr_info = item_dict['transcription']
                if 'a.m.' in ocr_info or 'a.m' in ocr_info or 'p.m.' in ocr_info or 'p.m.' in ocr_info:
                    pattern = r'\d{2}/\d{2}/\d{4} \d{2}:\d{2} [ap]\.m\.'
                    print("++++++++++++++++++++")
                    print(ocr_info)
                    print("++++++++++++++++++++")
                    match = re.search(pattern, ocr_info)
                    if match:
                        result = match.group()
                        print("++++++++++++++++++++")
                        print(result)
                        print("++++++++++++++++++++")
                        returned_value = result

    return returned_value

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
        aquisition_time = get_aquisition_time(image_path)
        image_id = int(os.path.splitext(file_name)[0])
        # 构造 JSON 注释内容
        annotation = {
            "image_id": image_id,
            "file_name": file_name,
            "image_width": image_width,
            "image_height": image_height,
            "aquisition_time": aquisition_time,
            "properties": {
                "Teeth": [],
                "Quadrants": [],
                "JawBones": []
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
