import json

# 文件路径
qa_path = "./dental_json/classification/xx/xx_qa.json"
dialogue_path = "./dental_json/classification/xx/xx_qa_dialogue823.json"
output_path = "./dental_json/classification/xx/xx_COT+dialogue_merged.json"

# 读取文件
with open(qa_path, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

with open(dialogue_path, "r", encoding="utf-8") as f:
    dialogue_data = json.load(f)

# 建立字典方便查找
qa_dict = {item["id"]: item for item in qa_data}
dialogue_dict = {item["id"]: item for item in dialogue_data}

# 提醒哪些 id 缺失
qa_ids = set(qa_dict.keys())
dialogue_ids = set(dialogue_dict.keys())

missing_in_dialogue = qa_ids - dialogue_ids
missing_in_qa = dialogue_ids - qa_ids

if missing_in_dialogue:
    print(f"⚠️ 以下 QA id 在 dialogue 中缺失: {missing_in_dialogue}")
if missing_in_qa:
    print(f"⚠️ 以下 dialogue id 在 QA 中缺失: {missing_in_qa}")

# 合并
merged_data = []
for _id, qa_item in qa_dict.items():
    dialogue_item = dialogue_dict.get(_id, {})
    merged_item = {
        "id": _id,
        "image": qa_item.get("image"),
        "question": qa_item.get("question"),
        "cot_answer": qa_item.get("cot_answer"),
        "diagnosis": dialogue_item.get("diagnosis"),
        "dialogue": dialogue_item.get("dialogue"),
        "Modality": qa_item.get("Modality"),
        "Disease category": qa_item.get("Disease category")
    }

    # 检查字段缺失
    missing_fields = [k for k, v in merged_item.items() if v in [None, ""]]
    if missing_fields:
        print(f"⚠️ id {_id} 缺失字段: {missing_fields}")

    merged_data.append(merged_item)

# 保存合并后的 JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"✅ 合并完成，共 {len(merged_data)} 条记录，保存至 {output_path}")
