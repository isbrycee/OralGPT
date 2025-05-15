import json
import os
from transformers import AutoTokenizer

def calculate_conversation_stats(folder_path):
    # 初始化分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        return

    total_tokens = 0
    total_messages = 0
    processed_files = 0

    # 遍历文件夹
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
        # 验证数据结构
        if "conversations" not in data:
            print(f"文件 {filename} 缺少 conversations 字段，已跳过")
            continue

        conversations = data["conversations"]
        if not isinstance(conversations, list):
            print(f"文件 {filename} 的 conversations 字段类型错误，已跳过")
            continue

        file_tokens = 0
        valid_messages = 0

        # 处理每个对话回合
        for turn in conversations:
            try:
                content = turn.get("content", "")
                if not isinstance(content, str):
                    print(f"文件 {filename} 存在非文本content，已跳过该对话回合")
                    continue
                
                tokens = len(tokenizer.encode(content, add_special_tokens=False))
                file_tokens += tokens
                valid_messages += 1
            except KeyError:
                print(f"文件 {filename} 存在格式不完整的对话回合")
                continue

        if valid_messages > 0:
            total_tokens += file_tokens
            total_messages += valid_messages
            processed_files += 1
            # print(f"已处理 {filename}: {valid_messages} 条对话 (共 {file_tokens} tokens)")

    # 输出统计结果
    print("\n===== 最终统计 =====")
    print(f"处理文件总数: {processed_files}")
    print(f"有效对话总数: {total_messages}")
    
    if total_messages > 0:
        avg_length = total_tokens / processed_files
        print(f"平均对话长度: {avg_length:.2f} tokens")
    else:
        print("没有找到有效对话内容")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python conversation_stats.py <json文件夹路径>")
        sys.exit(1)
    
    calculate_conversation_stats(sys.argv[1])