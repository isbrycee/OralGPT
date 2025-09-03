import os
import json
import base64
import time
import random
from openai import OpenAI

# ====== 配置 ======
client = OpenAI(api_key="sk-xx", base_url="https://api.chatanywhere.org/v1")

image_folder = ".Oral-GPT-data/image/OralGPT-RGB-Classification-Dataset/braces/"
output_json_path = "./Oral-GPT-data/dental_json/classification/braces/braces_qa.json"

# 10 个不同的问法
fixed_questions = [
    "This is an intraoral photograph of the oral cavity. Please provide a detailed description of its visible features.",
    "This is an intraoral photograph of the oral cavity. Give a comprehensive account of what is shown.",
    "This is an intraoral photograph of the oral cavity. How would you describe this image in detail?",
    "This is an intraoral photograph of the oral cavity. Provide a full description of the structures observed.",
    "This is an intraoral photograph of the oral cavity. Please generate a thorough description of the image.",
    "This is an intraoral photograph of the oral cavity. Write a detailed narrative of what can be seen.",
    "This is an intraoral photograph of the oral cavity. Offer a complete and careful description of the picture.",
    "This is an intraoral photograph of the oral cavity. Summarize the image with a detailed descriptive account.",
    "This is an intraoral photograph of the oral cavity. Please describe the image comprehensively.",
    "This is an intraoral photograph of the oral cavity. Generate a clear and detailed description of the photo."
]

## 不同分类 request 要修改
request = """ 
Output format: 
<Caption> Provide a detailed and clear description of the intraoral image, capturing its overall visual appearance. </Caption>

"""

max_retries = 3
sleep_between_requests = 2

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_cot(image_path, prompt):
    image_data = encode_image(image_path)
    messages = [
        {
            "role": "system",
           "content": (
            "You are a senior orthodontist. Your task is to generate a realistic and medically accurate description, "
            "focusing only on the observable findings in the image that are related to orthodontics. "
            "The image may correspond to one of three orthodontic stages: pre-treatment with misaligned teeth, "
            "in-treatment with braces or other orthodontic appliances, or post-treatment with well-aligned teeth. "
            "Ignore all non-oral facial features and focus exclusively on the teeth, lips, and mouth region. "
            "If you cannot confidently identify the image as a clear oral/orthodontic image, respond with: "
            "\"Unable to analyze: please take a clear photo of the teeth, lips, and mouth region.\" "
            "Otherwise, strictly follow this output format, including all opening and closing tags exactly as written. "
            "Never omit or misspell any tag. "
            "<Caption> ... </Caption>\n"
        )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages
    )
    return response.choices[0].message.content

# 读取已处理的数据
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    # ✅ 修复：只存文件名，避免跳过失败
    processed_images = {os.path.basename(entry["image"]) for entry in results}
else:
    results = []
    processed_images = set()

# 支持的图片后缀
supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# 获取所有支持的图像文件
image_files = [
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1].lower() in supported_exts
]

# 排序
image_files.sort()

# 如果想限制前几张处理，可以保留这一行
image_files = image_files[:130000]

for filename in image_files:
    if filename in processed_images:
        print(f"Skipping already processed: {filename}")
        continue

    image_path = os.path.join(image_folder, filename)

    # 每张图片随机选一个 fixed_question
    fixed_question = random.choice(fixed_questions)
    prompt = f"{fixed_question}\n\n{request}"

    for attempt in range(max_retries):
        try:
            cot_result = generate_cot(image_path, prompt)
            
            # 构建标准化结果
            entry = {
                "id": f"braces_diagnosis_{len(results) + 1}",  # 自动递增 id
                "image": os.path.join("OralGPT-RGB-Classification-Dataset/braces", filename),
                "question": fixed_question,
                "caption": cot_result,
                "Modality": "Intraoral photograph",
                "Disease category": "braces"
            }
            
            results.append(entry)
            processed_images.add(filename)  # ✅ 这里存的是纯文件名
            print(f"Processed: {filename}")
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {filename}: {e}")
            time.sleep(sleep_between_requests)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nAll done! Results saved to {output_json_path}")
