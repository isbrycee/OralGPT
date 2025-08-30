import json
import os

# 输入 .jsonl 文件路径
input_jsonl_path = "./dental_json/classification/cancer/cancer_qa_dialogue.jsonl"
# 输出标准 JSON 文件路径
output_json_path = "./dental_json/classification/cancer/cancer_qa_dialogue823.json"

# 检查输入文件是否存在
if not os.path.exists(input_jsonl_path):
    raise FileNotFoundError(f"{input_jsonl_path} 不存在")

# 读取 .jsonl 并转换为 list
data_list = []
with open(input_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            data_list.append(obj)
        except json.JSONDecodeError as e:
            print(f"解析失败，跳过这一行：{line[:100]}...，错误：{e}")
            continue

# 写入标准 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"✅ 已成功将 {len(data_list)} 条记录写入标准 JSON 文件：{output_json_path}")
