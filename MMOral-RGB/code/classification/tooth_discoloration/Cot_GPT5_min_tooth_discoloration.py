import os
import json
import base64
import time
import random
from openai import OpenAI

# ====== 配置 ======
client = OpenAI(api_key="sk-xx", base_url="https://api.chatanywhere.org/v1")

image_folder = "./image/OralGPT-RGB-Classification-Dataset/tooth_discoloration"
output_json_path = "./dental_json/classification/tooth_discoloration/tooth_discoloration_qa.json"

# 10 个不同的问法
fixed_questions = [
    "This is an intraoral photograph of the oral cavity. Please identify the disease present.",
    "This is an intraoral photograph of the oral cavity. What disease can you observe?",
    "This is an intraoral photograph of the oral cavity. Please determine the diagnosis.",
    "This is an intraoral photograph of the oral cavity. What condition is visible?",
    "This is an intraoral photograph of the oral cavity. Identify any oral diseases present.",
    "This is an intraoral photograph of the oral cavity. What is the disease present in this image?",
    "This is an intraoral photograph of the oral cavity. Please check for any oral disease.",
    "This is an intraoral photograph of the oral cavity. What is the likely diagnosis?",
    "This is an intraoral photograph of the oral cavity. Can you identify the condition shown?",
    "This is an intraoral photograph of the oral cavity. Which oral disease is depicted here?"
]
##不同分类request要修改
request = """ 
Output format: 
<Caption> Describe the intraoral image in detail, including the tooth and/or arch location, and the appearance of any tooth discoloration. Focus on features such as color changes (yellow, brown, gray, blue, or black), extent (localized or generalized), pattern (patchy, uniform, or banded), enamel or dentin involvement, and any associated surface changes like opacity, hypoplasia, or staining. </Caption>

<Think> 
1. Explain the visible features in the image that match the known characteristics of tooth discoloration.  
2. Recall the typical clinical appearance and causes of tooth discoloration from professional knowledge, including intrinsic and extrinsic factors.  
3. Confirm that the observed image appearance matches the knowledge-based visual representation of tooth discoloration, and rule out other dental conditions.  
</Think> 

<Answer> Summarize the matched disease(s)/condition(s) from step 3 of the above <Think> pipeline into one concise, fluent, professional medical diagnosis statement that clearly specifies the tooth discoloration type and affected teeth, without including any treatment recommendations or management advice. </Answer>
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
                "You are a senior dentist. Your task is to generate a realistic and medically accurate clinical reasoning summary, "
                "explaining the observable findings in the image and how they support the diagnosis. "
                "You must strictly follow this output format, including all opening and closing tags exactly as written. "
                "Never omit or misspell any tag."
                "<Caption> ... </Caption>\n"
                "<Think> ... </Think>\n"
                "<Answer> ... </Answer>\n"
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
    processed_images = {os.path.basename(entry["image"]) for entry in results}
else:
    results = []
    processed_images = set()

# 按数字排序
supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# 获取所有支持的图像文件
image_files = [
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1].lower() in supported_exts
]

# 排序
image_files.sort()

# 如果想限制前几张处理，可以保留这一行
image_files = image_files[:3000]

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
                "id": f"tooth_discoloration_diagnosis_{len(results) + 1}",  # 自动递增 id
                "image": os.path.join("OralGPT-RGB-Classification-Dataset/tooth_discoloration", filename),
                "question": fixed_question,
                "cot_answer": cot_result,
                "Modality": "Intraoral photograph",
                "Disease category": "tooth_discoloration"
            }
            
            results.append(entry)
            processed_images.add(filename)
            print(f"Processed: {filename}")
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {filename}: {e}")
            time.sleep(sleep_between_requests)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nAll done! Results saved to {output_json_path}")
