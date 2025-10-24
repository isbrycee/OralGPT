import os
import json

def count_json_list_lengths(folder_path: str):
    """
    只统计指定文件夹下的 JSON 文件（不递归子文件夹），
    按文件名排序后打印每个 JSON 文件中 list 的长度。
    """
    if not os.path.isdir(folder_path):
        print(f"路径无效或不是文件夹：{folder_path}")
        return

    results = []

    # 遍历当前文件夹下的文件（不递归）
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        results.append((file_name, len(data)))
                    else:
                        results.append((file_name, f"不是 list 格式 (类型: {type(data).__name__})"))
                except Exception as e:
                    results.append((file_name, f"读取失败（错误：{e}）"))

    # 按文件名排序
    results.sort(key=lambda x: x[0])

    # 打印结果
    for file_name, result in results:
        print(f"{file_name}: {result}")

if __name__ == "__main__":
    # 输入要扫描的文件夹路径
    folder = "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/"
    count_json_list_lengths(folder)
