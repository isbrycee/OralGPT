import os
import json
import base64
import random
from openai import OpenAI

# ====== 配置 ======
client = OpenAI(
    api_key="xx",
    base_url="https://api.chatanywhere.org/v1"
)

image_folder = "./Oral-GPT-data/image/MM-Oral-Periapical-images"
json_folder = "./Oral-GPT-data/dental_json/MM-Oral-Periapical-jsons-filter/"
output_json_path = "./Oral-GPT-data/dental_json/Periapical_generation/demo_each_category.json"

fixed_questions = [
   "This is a periapical X-ray image. Please identify the disease or condition.",
   "This is a periapical X-ray image. What disease or condition is present?",
   "This is a periapical X-ray image. What disease or condition can be observed?",
   "This is a periapical X-ray image. Please determine the disease or condition shown.",
   "This is a periapical X-ray image. What disease or condition does it suggest?",
   "This is a periapical X-ray image. Identify the underlying disease or condition.",
   "This is a periapical X-ray image. What disease or condition is visible here?",
   "This is a periapical X-ray image. Please specify the disease or condition demonstrated.",
   "This is a periapical X-ray image. What disease or condition can be diagnosed?",
   "This is a periapical X-ray image. State the most likely disease or condition."
]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_cot(image_path, prompt):
    image_data = encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a periapical X-ray image, and you may also be provided with a preliminary assessment that suggests possible diseases or conditions in the image. "
                "This preliminary assessment is for reference only and may be inaccurate. It may be provided at the image level or the region level. "
                "Your task is to generate a realistic and medically accurate clinical reasoning summary, explaining the observable findings in the image and how they support the diagnosis. "
                "Note that the preliminary assessment may be incorrect and may also miss additional diseases/conditions present in the image; carefully examine the visual features to reach the final conclusion. "
                "Do not attempt to predict whether a tooth is in the upper or lower jaw, and do not attempt to predict its specific type (e.g., second premolar) unless you are 100% certain."
                "You must strictly follow this output format, including all opening and closing tags exactly as written. Never omit or misspell any tag. Do not add any extra sections.\n\n"
                "<Caption> Provide a detailed description of the periapical X-ray image. </Caption>\n\n"
                "<Think>\n"
                "1. Analyze the visual features listed in <Caption> and determine which diseases or conditions they may correspond to.\n"
                "2. Recall the typical radiographic characteristics of the suspected diseases or conditions, including common variations and differential considerations.\n"
                "3. Compare the observed features with the typical characteristics, weigh alternative explanations, and decide on the most likely final diagnosis.\n"
                "Do not restate the instruction text above; instead, directly write your own reasoning content under each numbered step.\n"
                "</Think>\n\n"
                "<Answer> Based on the conclusion from Step 3 in <Think>, summarize the diagnosis in a single concise, fluent, and professional radiology report-style sentence. Avoid repeating the reasoning process here. </Answer>"
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
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

# ====== 遍历 JSON 并实时写入 ======
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
results = []
seen_entries = set()  # 存储已经生成的条目的 image_name 或 id
sample_count = 0
max_samples = 6000000  # 先生成6个

# 如果已经有生成的文件，读取已有条目
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        for entry in results:
            seen_entries.add(entry["image"])  # 用 image_name 判断是否已生成
        sample_count = len(results)

# ====== 遍历 JSON 并生成新样本 ======
for filename in os.listdir(json_folder):
    if sample_count >= max_samples:
        break
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, filename)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data.get("file_name")
    image_path = os.path.join(image_folder, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # ====== 如果该条目已生成，跳过 ======
    if image_name in seen_entries:
        print(f"Skipping already generated entry: {image_name}")
        continue

    # ====== 构建 Disease category 用的数据 ======
    classification_data = data["properties"].get("Classification", {})
    present_diseases = [name for name, info in classification_data.items() if info.get("present")] if classification_data else []

    location_data = data["properties"].get("Location", {})
    region_diseases = list(location_data.keys()) if location_data else []

    # ====== 构建用于 prompt 的 preliminary_assessment_text (去掉 "Pulpitis") ======
    prompt_diseases = [d for d in present_diseases if d != "Pulpitis"]

    preliminary_assessment = []
    for d in prompt_diseases:
        preliminary_assessment.append(f"Image-level disease or condition: {d}.")
    for loc_name in region_diseases:
        bbox = location_data[loc_name].get("bbox")
        if bbox:
            preliminary_assessment.append(f"Region-level disease or condition: {loc_name} at bbox {bbox}.")
    preliminary_assessment_text = "\n".join(preliminary_assessment)

    fixed_question = random.choice(fixed_questions)
    if preliminary_assessment_text.strip():
        prompt = f"{fixed_question}\nPreliminary assessment:\n{preliminary_assessment_text}"
    else:
        prompt = fixed_question

    # ====== 调用模型生成 CoT ======
    try:
        cot_result = generate_cot(image_path, prompt)

        # ====== 确定 Disease category (entry 用) ======
        if present_diseases:
            disease = present_diseases[0]  # 优先 Image-level
        elif region_diseases:
            disease = region_diseases[0]   # 没有 Image-level，取 Region-level
        else:
            disease = "Unknown"

        entry = {
            "id": f"demo_{disease}_{sample_count}",
            "image": image_name,
            "question": fixed_question,
            "cot_answer": cot_result,
            "Modality": "Periapical X-ray",
            "Disease category": disease
        }

        results.append(entry)
        seen_entries.add(image_name)
        sample_count += 1

        # ====== 实时写入 ======
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Generated demo {sample_count} for image: {image_name}")

    except Exception as e:
        print(f"Failed for {filename}: {e}")

print(f"\nDemo done! {sample_count} samples saved to {output_json_path}")