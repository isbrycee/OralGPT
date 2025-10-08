import os
import pandas as pd
import json
import re
import sys
from collections import Counter, defaultdict
import random

category_aliases = {
    "Impacted tooth": "Impacted tooth",
    "Pulpitis": "Pulpitis",
    "Deep_Caries": "Caries",
    "Caries": "Caries",
    "Periodontitis": "Periodontitis",
    "Apical periodontitis": "Apical periodontitis",
    "Bone loss": "Bone loss",
    "RCT": "Root canal treatment",
    "Crown": "Crown",
    "Restoration": "Restoration",
    "Alternation between primary and permanent teeth": "Mixed dentition",
}

def excel_to_json(excel_path, json_path="output.json", category_limits=None, random_seed=42):
    """
    category_limits: dict, e.g. {"braces": 100, "healthy": 200}
                     指定某些类别的最大保留数量
    random_seed: int, 固定随机数种子（保证每次运行结果一致）
    """
    random.seed(random_seed)
    
    # 读取 Excel 文件
    df = pd.read_excel(excel_path)
    
    # 检查倒数第二列
    target_col = df.columns[4]

    total_count = len(df)
    d_count = (df[target_col] == 'd').sum()
    one_count = (df[target_col] == 1).sum()
    
    print(f"总行数: {total_count}")
    print(f"倒数第二列为 'd' 的数量: {d_count}")
    print(f"倒数第二列为 '1' 的数量: {one_count}")

    records_tmp = defaultdict(list)  # 临时分组存储
    for _, row in df.iterrows():
        image_path_list = [str(row["图像路径"])]
        label = str(row[target_col]).strip()
        # print(label)
        # 跳过 label == 'd' 的行
        if label.lower() == 'd':
            continue

        # 获取类别: 根据图像路径倒数第二个元素
        image_path = str(row["图像路径"])
        category = str(row["疾病类别"])
        question = str(row["问题描述"])
        question = question.replace('disease or condition', 'disease(s) or condition(s)')
        
        answer_with_think = str(row["英文诊断结果"])
        answer = re.findall(r"<Answer>(.*?)</Answer>", answer_with_think, re.DOTALL)[0].strip()

        CoT = str(row["英文 caption"]) + ' </Caption>\n' + answer_with_think
        
        record = {
            "conversations": [
                {
                    "from": "human",
                    "value": "<image> " + question
                },
                {
                    "from": "gpt",
                    "value": answer,
                    "value_with_CoT": CoT
                }
            ],
            "system": "",
            "category": 'PA,'+category_aliases[category],
            "images": image_path_list
        }
        records_tmp[category].append(record)
        
    # 应用类别控制（随机下采样）
    final_records = []
    for category, items in records_tmp.items():
        if category_limits and category in category_limits:
            max_num = category_limits[category]
            if len(items) > max_num:
                sampled = random.sample(items, max_num)
                final_records.extend(sampled)
                print(f"类别 {category}: 原始 {len(items)} 条，下采样至 {max_num} 条")
            else:
                final_records.extend(items)
                print(f"类别 {category}: {len(items)} 条（未触及上限）")
        else:
            final_records.extend(items)

    # 保存为 json 文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=4, ensure_ascii=False)

    print(f"已保存为 {json_path}")

    # 类别统计
    category_counter = Counter([rec["category"] for rec in final_records])
    print("\n=== 有效数据类别分布（保存后） ===")
    for cat, count in category_counter.items():
        print(f"{cat}: {count}")

if __name__ == "__main__":
    excel_path = '/home/jinghao/projects/x-ray-VLM/RGB/periapical_radiographs/beenchmark_excel_for_dentist_validation_periapical_image_for_diagnosis.xlsx'
    json_path = '/home/jinghao/projects/x-ray-VLM/RGB/periapical_radiographs/beenchmark_excel_for_dentist_validation_periapical_image_for_diagnosis_ShareGPT_format.json'

    # 在这里自定义 braces / healthy 的数量上限
    limits = {
        "braces": 70,   # 最多保留 100 条
        "healthy": 100   # 最多保留 200 条
    }

    excel_to_json(excel_path, json_path, category_limits=limits, random_seed=42)