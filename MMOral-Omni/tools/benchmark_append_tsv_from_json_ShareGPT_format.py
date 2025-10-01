import json
import csv
import sys
csv.field_size_limit(sys.maxsize)

def json_to_tsv(json_path, tsv_path, output_path=None):
    # 如果没有指定输出，覆盖原 tsv 文件
    assert output_path is not None, "请指定 output_path，避免覆盖原文件"

    # 读取已有 tsv，找出最后一个 index
    last_index = 0
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            last_index = max(last_index, int(row["index"]))

    # 读取 json 文件
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 追加新数据
    new_rows = []
    for item in json_data:
        conversations = item.get("conversations", [])
        category = item.get("category", "")
        question = ""
        answer = ""
        for conv in conversations:
            if conv.get("from") == "human":
                question = conv.get("value", "").replace("\n", "\\n")
            elif conv.get("from") == "gpt":
                answer = conv.get("value", "").replace("\n", "\\n")
        last_index += 1
        new_rows.append({
            "index": str(last_index),
            "image": "",
            "question": question,
            "answer": answer,
            "category": category
        })

    # 合并原数据和新增数据，写回 TSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["index", "image", "question", "answer", "category"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_ALL )  # 关键设置：强制所有字段加双引号
        
        writer.writeheader()
        writer.writerows(rows + new_rows)

    print(f"已生成输出 TSV 文件: {output_path}")

# 示例调用
json_to_tsv(json_path="/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/benchmark_examination-k-2025.9.23-translate-J-2025.9.25.json", 
            tsv_path="/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_Treatment_Planning.tsv", 
            output_path="/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral.tsv")
