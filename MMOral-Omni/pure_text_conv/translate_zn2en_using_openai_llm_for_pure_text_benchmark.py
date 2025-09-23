import pandas as pd
import json
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-N1hsISExwkdoyisZg9gTd8CxzNAwK8r2ESRSbFsp2M2859Q6",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)


def translate_question(text, client):
    """
    将中文问题翻译成英文考试风格问句
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    """
                    You are a professional medical exam question writer and translator.
                    Translate the following Chinese text into fluent, natural English.
                    Always reframe the text into a clear exam-style question, even if the original is just a topic, phrase, or statement.
                    Ensure the output is precisely in the form of a question, written in an academic and professional medical examination style.
                    """
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def translate_answer(text, client):
    """
    将中文答案翻译成自然流畅的英文回答
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional translator.\n"
                    "Translate the following Chinese text into fluent and natural English.\n"
                    "Do not reframe the style, just provide a clear and accurate translation."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def process_excel(file_path, output_json):
    # 读取所有 sheet
    xls = pd.ExcelFile(file_path)
    all_data = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        for _, row in df.iterrows():
            question = str(row.iloc[0]).strip()
            answer = str(row.iloc[1]).strip()
            # 翻译
            question_en = translate_question(question, client)
            answer_en = translate_answer(answer, client)

            conversation_entry = {
                "conversations": [
                    {
                        "from": "human",
                        "value": question_en
                    },
                    {
                        "from": "gpt",
                        "value": answer_en
                    }
                ],
                "system": "You are a helpful assistant.",
                "category": sheet_name  # sheet 名也翻译成英文
            }
            all_data.append(conversation_entry)

    # 保存为 JSON 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/benchmark_examination-k-2025.9.23.xlsx"   # 输入 excel 文件
    output_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/benchmark_examination-k-2025.9.23.json" # 输出 json 文件
    process_excel(input_file, output_file)
    print(f"✅ 已经翻译完成并保存到 {output_file}")
