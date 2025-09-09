import os
import json

def merge_to_jsonl(folder_path, output_file="merged.jsonl"):
    with open(os.path.join(folder_path, output_file), "w", encoding="utf-8") as out_f:
        total = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                total += 1
                            else:
                                print(f"⚠️ 文件 {filename} 第 {line_num} 行不是字典，已跳过")
                        except json.JSONDecodeError:
                            print(f"⚠️ 文件 {filename} 第 {line_num} 行不是合法 JSON，已跳过")

    print(f"✅ 合并完成，共 {total} 条数据，保存到 {os.path.join(folder_path, output_file)}")


if __name__ == "__main__":
    folder = '/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_shareGPT'
    output_file = os.path.join(folder, "merged_all_textbook_english_shareGPT.jsonl")
    merge_to_jsonl(folder, output_file)
