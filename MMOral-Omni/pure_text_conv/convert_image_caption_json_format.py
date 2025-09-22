import json

def convert_format(input_file, output_file):
    # 读取输入 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    for item in data:
        conversations = []
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                conversations.append({
                    "from": "human",
                    "value": msg.get("content")
                })
            elif msg.get("role") == "assistant":
                conversations.append({
                    "from": "gpt",
                    "value": msg.get("content")
                })

        output.append({
            "conversations": conversations,
            "images": item.get("images", [])
        })

    # 保存输出 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/en_image_pair/en_image_formated.json"   # 你的输入文件
    output_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/en_image_pair/image_pair_formated_shareGPT_for_image_caption_alignment.json" # 输出文件名
    convert_format(input_file, output_file)
