import os
import json

def merge_json_files(input_folder, output_file):
    merged_data = []

    # 遍历目录下的 .json 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        print(f"⚠️ 文件 {filename} 不是 list，已跳过")
                except json.JSONDecodeError as e:
                    print(f"⚠️ 文件 {filename} JSON 解析失败: {e}")

    # 保存合并后的结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已合并 {input_folder} 下的 JSON 文件，结果保存在 {output_file}")

# 示例用法
if __name__ == "__main__":
    input_folder = "/home/jinghao/projects/x-ray-VLM/OralGPT/MMOral-Omni/cephalometric_radiographs"   # 替换为你的文件夹路径
    output_file = "merged.json"                 # 输出文件路径
    merge_json_files(input_folder, output_file)
