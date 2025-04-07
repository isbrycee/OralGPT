import json
import os
from docx import Document

# 配置路径
input_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/for_human_evaluation/jsons'      # 存放JSON文件的文件夹
output_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/for_human_evaluation/words'    # 存放Word文件的文件夹（自动创建）
target_field = 'med_report'         # 要提取的JSON字段名

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的JSON文件
for filename in os.listdir(input_folder):
    if not filename.endswith('.json'):
        continue  # 跳过非JSON文件
    
    input_path = os.path.join(input_folder, filename)
    output_name = os.path.splitext(filename)[0] + '.docx'  # 保留原文件名
    output_path = os.path.join(output_folder, output_name)
    
    try:
        # 读取JSON文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取字段内容
        content = data.get(target_field, '')
        if not content:
            print(f'跳过 {filename}: 未找到字段 "{target_field}"')
            continue
        
        # 处理转义字符（如 \n 被存储为 \\n 的情况）
        content = content.replace('\\n', '\n')
        
        # 生成Word文档
        doc = Document()
        for line in content.split('\n'):
            doc.add_paragraph(line)
        doc.save(output_path)
        
        print(f'已生成: {output_path}')
    
    except Exception as e:
        print(f'处理 {filename} 失败: {str(e)}')
