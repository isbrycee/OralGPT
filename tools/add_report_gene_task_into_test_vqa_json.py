import os
import json
import random

question_templates = ['Can you provide a caption consists of findings for this panoramic X-ray image?',
'Describe the findings of the panoramic X-ray image you see.',
'Please caption this panoramic X-ray scan with findings',
'What is the findings of this panoramic X-ray image?',
'Describe this panoramic X-ray scan with findings.',
'Please write a caption consists of findings for this panoramic X-ray image.',
'Can you summarize with findings the panoramic X-ray images presented?',
'Please caption this image with findings.',
'Please provide a caption consists of findings for this panoramic X-ray image.',
'Can you provide a summary consists of findings of this panoramic radiograph?',
'What are the findings presented in this panoramic X-ray scan?',
'Please write a caption consists of findings for this image.',
'Can you provide a description consists of findings of this panoramic X-ray scan?',
'Please caption this panoramic X-ray scan with findings.',
'Can you provide a caption consists of findings for this panoramic X-ray scan?',]

def update_json_files(source_folder, target_folder):
    # 遍历源文件夹，加载每个 JSON 文件
    for source_file in os.listdir(source_folder):
        source_path = os.path.join(source_folder, source_file)
        
        # 确保文件是 JSON 文件
        if source_file.endswith(".json") and os.path.isfile(source_path):
            with open(source_path, 'r', encoding='utf-8') as src_file:
                source_data = json.load(src_file)
            
            # 确保源文件中有 "med_report" 字段
            if "med_report" not in source_data:
                print(f"Skipping {source_file}: 'med_report' not found in source.")
                continue
            
            med_report = source_data["med_report"]
            
            # 在目标文件夹中找到对应的文件
            target_path = os.path.join(target_folder, source_file)
            if os.path.exists(target_path) and target_path.endswith(".json"):
                with open(target_path, 'r+', encoding='utf-8') as tgt_file:
                    target_data = json.load(tgt_file)
                    
                    # 确保目标文件中有 "vqa_data"
                    if "vqa_data" not in target_data:
                        print(f"Skipping {source_file}: 'vqa_data' not found in target.")
                        continue
                    
                    # 将数据插入到 "report_generation" 字段中
                    report_generation = target_data["vqa_data"].get("report_generation", [])
                    report_generation.append({
                        "Question": random.choice(question_templates),
                        "Answer": med_report,
                        "Category": "report"
                    })
                    target_data["vqa_data"]["report_generation"] = report_generation
                    
                    # 将更新后的 JSON 写回文件
                    tgt_file.seek(0)
                    json.dump(target_data, tgt_file, ensure_ascii=False, indent=4)
                    tgt_file.truncate()
                print(f"Updated {target_path} successfully.")
            else:
                print(f"Target file {source_file} not found in target folder.")

# 替换为实际的文件夹路径
source_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_100/MM-Oral-OPG-loc-med-reports"
target_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_100/MM-Oral-OPG-vqa-loc-med"

update_json_files(source_folder, target_folder)
