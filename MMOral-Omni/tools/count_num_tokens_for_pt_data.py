import json
import tiktoken
from tqdm import tqdm

def count_tokens_in_jsonl(jsonl_path, model_name="gpt-4"):
    """
    统计一个 JSONL 文件中所有文本的 token 数量

    参数:
        jsonl_path: JSONL 文件路径
        model_name: 使用的模型名称（决定 tokenizer 类型）
    """
    # 创建 tokenizer
    enc = tiktoken.encoding_for_model(model_name)

    total_tokens = 0
    num_texts = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            text = data.get("text", "")
            tokens = enc.encode(text)
            total_tokens += len(tokens)
            num_texts += 1

    avg_tokens = total_tokens / num_texts if num_texts > 0 else 0
    print(f"总文本数量: {num_texts}")
    print(f"总 token 数量: {total_tokens}")
    print(f"平均每条文本 token 数: {avg_tokens:.2f}")

    return total_tokens, avg_tokens


if __name__ == "__main__":
    # 示例用法
    jsonl_path = "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/1.1_pt_plaintext_textbooks16_mainlandCN_shareGPT.jsonl"  # 替换为你的 JSONL 文件路径
    count_tokens_in_jsonl(jsonl_path)
