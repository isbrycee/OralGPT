import json
import random
import argparse
import os

def split_json(input_file, val_count, seed=42, output_dir="."):
    # 设置随机种子，保证结果可复现
    random.seed(seed)

    # 读取原始数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入的 JSON 文件必须是一个 list，每个元素是一个 dict。")

    if val_count > len(data):
        raise ValueError("验证集数量不能大于数据总量。")

    # 随机打乱并划分
    indices = list(range(len(data)))
    random.shuffle(indices)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    val_data = [data[i] for i in val_indices]
    train_data = [data[i] for i in train_indices]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存划分后的数据
    train_file = os.path.join(output_dir, "train.json")
    val_file = os.path.join(output_dir, "val.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"训练集保存到: {train_file} (数量: {len(train_data)})")
    print(f"验证集保存到: {val_file} (数量: {len(val_data)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机划分 JSON 数据为训练/验证集")
    parser.add_argument("--input", type=str, default="/home/jinghao/projects/x-ray-VLM/RGB/periapical_radiographs/MMOral_V2_periapical_radiographs.json", help="输入 JSON 文件路径")
    parser.add_argument("--val_count", type=int, default=550, help="验证集数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认=42)")
    parser.add_argument("--out", type=str, default="/home/jinghao/projects/x-ray-VLM/RGB/periapical_radiographs/", help="输出目录 (默认=当前目录)")
    args = parser.parse_args()

    split_json(args.input, args.val_count, args.seed, args.out)
