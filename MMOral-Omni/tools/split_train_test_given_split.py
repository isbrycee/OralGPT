import json
import os
import sys

def split_json_by_field(json_path):
    """
    根据每个列表元素中的 split 字段，将 JSON 数据划分为 train 和 test 集合。
    保存为 {原文件名}_train.json 和 {原文件名}_test.json
    """
    # 读取原始 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查数据是否为 list 类型
    if not isinstance(data, list):
        raise ValueError("输入的 JSON 文件内容必须是一个 list")

    # 根据 split 字段划分数据
    train_data = [item for item in data if str(item.get("split", "")).lower() in ("train", "training", "validation", "val", "valid")]
    test_data = [item for item in data if str(item.get("split", "")).lower() in ("test")]

    # 生成新的文件名
    base, ext = os.path.splitext(json_path)
    train_path = f"{base}_train{ext}"
    test_path = f"{base}_test{ext}"

    # 保存为新的 JSON 文件
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"数据划分完成：\nTrain 集合保存至: {train_path}\nTest 集合保存至: {test_path}")
    print(f"Train 数量: {len(train_data)}，Test 数量: {len(test_data)}")

if __name__ == "__main__":
    split_json_by_field("/home/jinghao/projects/x-ray-VLM/RGB/intraoral_video_for_comprehension/6.1_intraoralVideo_Comprehension.json")
