import json
from openai import OpenAI
from tqdm.asyncio import tqdm  # 注意用异步版本的 tqdm
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(
    api_key="sk-",  # 替换成你的 DMXapi 令牌key
    base_url="http://localhost:8080/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)

# 系统提示，可以根据需要修改
SYSTEM_PROMPT = "You are a professional assistant for understanding text content and translating Chinese into Chinese."

# 用户任务提示（英文版）
USER_PROMPT = """
Translate the provided Chinese text into fluent, professional English. The source material is drawn from the book Oral and Maxillofacial Imaging Diagnostics and may contain technical terms and complex sentence structures. Follow these requirements carefully:

1. Accuracy & Tone: Convey the meaning precisely, maintaining an academic and professional style.
2. Relevance: Ignore content unrelated to oral mucosal pathology (e.g., mentions of images, tables, prefaces, publishing details, or WeChat public account).
3. Coherence: Reconstruct or adjust incomplete or fragmented sentences so the translation is smooth, logical, and easy to read.
4. Error Correction: Fix typographical mistakes or non-standard phrasing in the Chinese text, guided by context.
5. Output Format: Return only the translated English text, with no explanations, notes, or extra commentary.
"""


def translate_file(input_file, output_file):

    # 读取输入 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取待翻译的文本
    texts = [item["text"] for item in data if "text" in item]

    results = []
    for txt in texts:
        # 调用 API，这里用 responses 接口 (与 Chat 类似，但更通用)
        response = client.chat.completions.create(
            model="/data/llm_models/neuralmagic_deepseek/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16",
            messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        {"role": "user", "content": txt},
                    ],
        )

        # API 输出
        translated = response.choices[0].message.content.split('</think>')[1].strip()
        results.append({"text": translated})
        print(f"原文: {txt}\n\n{translated}\n{'-'*40}")

    # 保存为新的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"完成翻译，共 {len(results)} 条结果，保存到 {output_file}")

# 使用方式示例：
translate_file("/home/jinghao/projects/x-ray-VLM/OralGPT/MMOral-Omni/pure_text_conv/textbook_Oral_and_Maxillofacial_Imaging_Diagnostics.json", 
               "textbook_Oral_and_Maxillofacial_Imaging_Diagnostics_English_for_pretraining.json")