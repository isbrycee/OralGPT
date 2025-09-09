import json
from transformers import AutoTokenizer

# 选择一个适合的模型 tokenizer，例如 GPT-2
# 你也可以换成 "bert-base-chinese" 或其他 HuggingFace 模型
tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)

input_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_shareGPT/merged_all_textbook_english_shareGPT.jsonl"
output_file = "data_with_tokens.jsonl"

max_token_count = 0
max_text = None

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        data = json.loads(line)
        text = data.get("text", "")

        # 计算 token 数
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        # 保存到结果
        data["token_count"] = token_count
        f_out.write(json.dumps({"token_count": token_count}, ensure_ascii=False) + "\n")

        # 更新最大值
        if token_count > max_token_count:
            max_token_count = token_count
            max_text = text

print(f"✅ 已处理完成，结果已保存到 {output_file}")
print(f"最大 token 数: {max_token_count}")
print(f"对应文本: {max_text[:]}...")  # 只显示前 100 个字符，避免太长
