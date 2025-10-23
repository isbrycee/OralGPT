import json
import sys
from pathlib import Path
import tiktoken

def count_tokens(text, encoding_name="cl100k_base"):
    """
    使用 tiktoken 计算文本 token 数
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def analyze_json_file(file_path: str):
    """
    对单个 JSON 文件进行统计分析
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"文件 {file_path} 内容必须是一个 list 类型")

    total_human_tokens = 0
    total_gpt_tokens = 0
    total_tokens_all_dicts = 0

    for item in data:
        conversations = item.get("conversations", [])
        human_tokens = 0
        gpt_tokens = 0

        for conv in conversations:
            text = conv.get("value", "")
            if conv.get("from") == "human":
                human_tokens += count_tokens(text)
            elif conv.get("from") == "gpt":
                gpt_tokens += count_tokens(text)

        total_human_tokens += human_tokens
        total_gpt_tokens += gpt_tokens
        total_tokens_all_dicts += (human_tokens + gpt_tokens)

    avg_tokens_per_dict = total_tokens_all_dicts / len(data) if data else 0

    return {
        "file": file_path,
        "json_list_length": len(data),
        "human_token_total": total_human_tokens,
        "gpt_token_total": total_gpt_tokens,
        "average_tokens_per_dict": avg_tokens_per_dict
    }


def main(json_files):
    """
    支持多个 JSON 文件批量计算
    """
    results = []
    for file_path in json_files:
        stats = analyze_json_file(file_path)
        results.append(stats)
    
    # 输出总体统计结果
    print("=" * 80)
    print("统计结果：\n")
    for res in results:
        print(f"文件名: {res['file']}")
        print(f"- JSON list 长度: {res['json_list_length']}")
        print(f"- Human token 总数: {res['human_token_total']}")
        print(f"- GPT token 总数: {res['gpt_token_total']}")
        print(f"- 每个 dict 平均 token 数: {res['average_tokens_per_dict']:.2f}")
        print("-" * 80)
    
    # 计算所有文件的汇总信息
    total_len = sum(r["json_list_length"] for r in results)
    total_human = sum(r["human_token_total"] for r in results)
    total_gpt = sum(r["gpt_token_total"] for r in results)
    total_token = total_human + total_gpt
    avg_token_per_dict = total_token / total_len if total_len else 0

    print("\n总体汇总：")
    print(f"- 所有文件的 JSON list 总长度: {total_len}")
    print(f"- Human token 总数: {total_human}")
    print(f"- GPT token 总数: {total_gpt}")
    print(f"- 平均每个 dict token 数: {avg_token_per_dict:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    

    json_files = [
        "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/0.1_sft_multimodal_ALL_shareGPT.json",
        "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/6.1_sft_intraoralVideo_Comprehension_Vident_shareGPT.json",
        "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/9.1_sft_panoramicImage_ReportVQAChat_MMOralOPG_1.3w_shareGPT.json",
    ]
    main(json_files)
