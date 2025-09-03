import json
import os
import random
from pathlib import Path

# 配置路径
json_dir = Path("./oral-classification-all-json")
train_dir = json_dir / "train"
test_dir = json_dir / "test"

# 创建输出文件夹
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# 设置随机种子保证可复现
random.seed(42)

# 遍历 json 文件
for json_file in json_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 打乱顺序
    random.shuffle(data)

    # 按 15% 划分
    split_idx = int(len(data) * 0.15)
    test_data = data[:split_idx]
    train_data = data[split_idx:]

    # 保存到对应文件夹
    train_path = train_dir / json_file.name
    test_path = test_dir / json_file.name

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已处理 {json_file.name}: train {len(train_data)}条, test {len(test_data)}条")
