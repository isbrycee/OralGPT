import os
import json
from collections import OrderedDict

def add_field_to_jsons(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    num = 0
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_quality_bool = data['image_width'] == data['image_height']
                if image_quality_bool:
                    image_quality = 'low'
                    num += 1
                else:
                    image_quality = 'high'
                if isinstance(data, dict):  # 单个 JSON 对象
                    new_data = OrderedDict()
                    for key, value in data.items():
                        new_data[key] = value
                        if key == "file_name":  # 在 file_name 后插入 image_modality
                            new_data["image_modality"] = 'Panoramic X-ray'
                        if key == "image_height":  # 在 file_name 后插入 image_modality
                            new_data["image_quality"] = image_quality
                    if "image_modality" not in new_data:  # 如果 file_name 不存在，则直接添加
                        new_data["image_quality"] = image_quality
                    data = new_data
                
                elif isinstance(data, list):  # JSON 数组
                    for item in data:
                        if isinstance(item, dict):
                            new_item = OrderedDict()
                            for key, value in item.items():
                                new_item[key] = value
                                if key == "file_name":
                                    new_item["image_modality"] = "aaa"
                            if "image_modality" not in new_item:
                                new_item["image_modality"] = "aaa"
                            item.clear()
                            item.update(new_item)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                print(f"处理成功: {filename}")
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    print(num)

# 使用示例
input_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/refined_report_4o'  # 替换为你的输入文件夹路径
output_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/MM-Oral-OPG-jsons-loc-med-report'  # 替换为你的输出文件夹路径

add_field_to_jsons(input_folder, output_folder)
