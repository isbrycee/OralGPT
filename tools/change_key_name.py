import os
import json

# 指定你的 JSON 文件夹路径
folder_path = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/train/MM-Oral-OPG-vqa-loc-med"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # 确保只处理 JSON 文件
        file_path = os.path.join(folder_path, filename)
        
        # 打开并读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 遍历 JSON 数据并修改字段
        def update_category(obj):
            if isinstance(obj, dict):
                if 'explanation' in obj:
                    obj['Explanation'] = obj.pop('explanation')  # 修改字段名
                for key in obj:
                    update_category(obj[key])  # 递归处理嵌套结构
            elif isinstance(obj, list):
                for item in obj:
                    update_category(item)  # 递归处理列表内的元素
        
        update_category(data)
        
        # 将修改后的 JSON 写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        # print(f"已处理文件: {filename}")

print("所有 JSON 文件的 category 字段已成功修改为 Category。")
