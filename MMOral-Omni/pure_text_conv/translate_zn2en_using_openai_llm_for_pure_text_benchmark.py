import pandas as pd
import json
import os
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="sk-1",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)


def translate_question(text, client, max_retries=10):
    """
    将中文问题翻译成英文考试风格问句
    如果 response 返回 None 或出错，则自动尝试重新生成
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional medical exam question writer and translator.\n"
                            "Translate the following Chinese text into fluent, natural English.\n"
                            "Always reframe the text into a clear exam-style question, even if the original is just a topic, phrase, or statement.\n"
                            "Ensure the output is precisely in the form of a question, written in an academic and professional medical examination style."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0
            )

            # 确保 response 内容合法
            if (
                response is not None
                and hasattr(response, "choices")
                and len(response.choices) > 0
                and hasattr(response.choices[0], "message")
                and response.choices[0].message is not None
                and hasattr(response.choices[0].message, "content")
            ):
                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[translate_question] Attempt {attempt + 1} failed: {e}")

    # 如果尝试多次都失败，返回空字符串
    return ""


def translate_answer(text, client, max_retries=10):
    """
    将中文答案翻译成自然流畅的英文回答
    如果 response 返回 None 或出错，则自动尝试重新生成
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    for attempt in range(max_retries):
        try:
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

            # 确保 response 和必要字段存在
            if (
                response is not None
                and hasattr(response, "choices")
                and len(response.choices) > 0
                and hasattr(response.choices[0], "message")
                and response.choices[0].message is not None
                and hasattr(response.choices[0].message, "content")
            ):
                return response.choices[0].message.content.strip()

        except Exception as e:
            # 可以打印错误日志，方便排查
            print(f"[translate_answer] Attempt {attempt + 1} failed: {e}")

    # 如果多次尝试仍然失败，返回空字符串或提示信息
    return ""

def process_excel(file_path, output_json):
    # 读取所有 sheet
    xls = pd.ExcelFile(file_path)
    all_data = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        for _, row in tqdm(df.iterrows()):
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
            print(conversation_entry)
            
    # 保存为 JSON 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/benchmark_examination-k-2025.9.23.xlsx"   # 输入 excel 文件
    output_file = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/benchmark_examination-k-2025.9.23.json" # 输出 json 文件
    process_excel(input_file, output_file)
    print(f"✅ 已经翻译完成并保存到 {output_file}")
