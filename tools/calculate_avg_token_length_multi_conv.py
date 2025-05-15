import json
import os
from transformers import AutoTokenizer

def process_options(options):
    """处理不同格式的选项数据"""
    if isinstance(options, dict):
        return " ".join([f"{k}: {v}" for k, v in options.items()])
    if isinstance(options, str):
        return options
    return ""

def calculate_vqa_stats(folder_path):
    # 初始化分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return

    total_tokens = 0
    total_questions = 0
    processed_files = 0
    qa_fields = [
        "loc_closed_ended", 
        "loc_open_ended",
        "med_closed_ended",
        "med_open_ended"
    ]

    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"无法解析文件 {filename}: {e}")
            continue

        # 提取vqa_data
        vqa_data = data.get("vqa_data", {})
        if not isinstance(vqa_data, dict):
            print(f"文件 {filename} 的vqa_data格式错误，已跳过")
            continue

        file_tokens = 0
        valid_questions = 0

        # 遍历所有QA字段
        for field in qa_fields:
            questions = vqa_data.get(field, [])
            if not isinstance(questions, list):
                continue

            # 处理每个问题
            for qa in questions:
                try:
                    # 构建完整问题文本
                    question = qa.get("Question", "")
                    options = qa.get("Options", "")
                    
                    # 合并选项内容
                    option_text = process_options(options)
                    full_text = f"{question} {option_text}".strip()
                    
                    # 计算token长度
                    tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
                    
                    file_tokens += tokens
                    valid_questions += 1
                except Exception as e:
                    print(f"处理 {filename} 的问题时出错: {e}")
                    continue

        if valid_questions > 0:
            total_tokens += file_tokens
            total_questions += valid_questions
            processed_files += 1
            print(f"已处理 {filename}: {valid_questions} 个问题 (共 {file_tokens} tokens)")

    # 输出统计结果
    print("\n===== 最终统计 =====")
    print(f"处理文件总数: {processed_files}")
    print(f"有效问题总数: {total_questions}")
    
    if total_questions > 0:
        avg_length = total_tokens / total_questions
        print(f"平均问题长度: {avg_length:.2f} tokens")
    else:
        print("没有找到有效问题")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python vqa_stats.py <json文件夹路径>")
        sys.exit(1)
    
    calculate_vqa_stats(sys.argv[1])
