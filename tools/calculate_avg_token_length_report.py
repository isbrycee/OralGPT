import json
import os
from transformers import AutoTokenizer

def process_folder(folder_path):
    # 初始化分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        print("请确认您已安装transformers库并通过huggingface-cli登录")
        return

    total_loc, total_med, file_count, entry_count = 0, 0, 0, 0
    total_all = 0
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
        
        # 统一数据结构
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            print(f"文件 {filename} 包含无效格式，已跳过")
            continue
        
        # 处理每个条目
        valid_entries = 0
        for entry in data:
            try:
                loc = str(entry['loc_caption'])  # 强制类型转换
                med = str(entry['med_report'])
            except (KeyError, TypeError) as e:
                print(f"文件 {filename} 存在字段缺失或类型错误: {e}")
                continue
            
            # 计算token长度
            loc_len = len(tokenizer.encode(loc, add_special_tokens=False))
            med_len = len(tokenizer.encode(med, add_special_tokens=False))
            
            total_loc += loc_len
            total_med += med_len
            total_all += loc_len
            total_all += med_len
            valid_entries += 1
        
        if valid_entries > 0:
            file_count += 1
            entry_count += valid_entries
            print(f"已处理 {filename} ({valid_entries} 条有效记录)")

    # 输出统计结果
    if entry_count == 0:
        print("没有找到有效数据")
        return
    
    print(f"\n处理完成：")
    print(f"- 共处理 {file_count} 个文件")
    print(f"- 共分析 {entry_count} 条有效记录")
    print(f"loc_caption平均token长度: {total_loc/entry_count:.2f}")
    print(f"med_report平均token长度: {total_med/entry_count:.2f}")
    print(f"both of them 平均token长度: {total_all/entry_count/2:.2f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python calculate_avg_token_length.py <json文件夹路径>")
        sys.exit(1)
    process_folder(sys.argv[1])
