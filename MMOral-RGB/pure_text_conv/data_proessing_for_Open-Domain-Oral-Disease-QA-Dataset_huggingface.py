import os
import json

# 输入输出文件路径
input_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/Open-Domain-Oral-Disease-QA-Dataset-huggingface/extracted_all.jsonl"   # 你的原始 jsonl 文件
output_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/Open-Domain-Oral-Disease-QA-Dataset-huggingface/extracted_all_sharedGPT.json"  # 转换后的 json 文件

data_out = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        record = json.loads(line)

        # 跳过 validity 为 incorrect 的数据
        if record.get("validity") != "correct":
            continue

        # 构造新格式
        new_entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": record.get("question", "")
                },
                {
                    "from": "gpt",
                    "value": record.get("Answer", "")
                }
            ],
            "title": record.get("disease", ""),
            "system": ""  # 可选字段，这里留空
        }

        data_out.append(new_entry)

# 保存为 json 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_out, f, ensure_ascii=False, indent=2)

print(f"转换完成！共保存 {len(data_out)} 条数据到 {output_file}")