import os
import json

def merge_json_lists(input_folder, output_file):
    merged_list = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            if "case_studies_conv" in filename:
                print(f"跳过文件 {filename}，因为包含 'case_studies_conv'")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(len(data), filename)
                        for item in data:
                            if isinstance(item, dict):
                                merged_list.append(item)
                            elif isinstance(item, list):
                                if len(item) == 1:
                                    print(item[0])
                                    merged_list.append(item[0])
                                else:
                                    merged_list.extend(item)
                    else:
                        print(f"⚠️ 文件 {filename} 不是列表，跳过")
                except json.JSONDecodeError as e:
                    print(f"❌ 解析失败 {filename}: {e}")
    
    # 保存合并后的列表
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(merged_list, out_f, ensure_ascii=False, indent=2)
    
    print(f"✅ 合并完成，保存到 {output_file}")

if __name__ == "__main__":
    input_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown"  # 输入你的文件夹路径
    output_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/hku_six_textbooks_QA_pairs_for_sft.json"        # 输出文件
    merge_json_lists(input_folder, output_file)
