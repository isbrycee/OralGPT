import os
import json

def merge_json_lists(input_folder, output_file):
    merged_list = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_list.extend(data)
                    else:
                        print(f"⚠️ 文件 {filename} 不是列表，跳过")
                except json.JSONDecodeError as e:
                    print(f"❌ 解析失败 {filename}: {e}")
    
    # 保存合并后的列表
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(merged_list, out_f, ensure_ascii=False, indent=2)
    
    print(f"✅ 合并完成，保存到 {output_file}")

if __name__ == "__main__":
    input_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/dental_QA_huggingface_from_dental_forum/sharGPT_data"  # 输入你的文件夹路径
    output_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/dental_QA_huggingface_from_dental_forum/dental_QA_from_dental_forum.json"        # 输出文件
    merge_json_lists(input_folder, output_file)
