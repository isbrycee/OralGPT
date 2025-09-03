import os
import json
import re
from openai import OpenAI

# ====== 配置 ======
client = OpenAI(
    api_key="sk-xx",
    base_url="https://api.chatanywhere.org/v1"
)

input_json_path = "./dental_json/classification/calculus/calculus_qa_updated.json"
output_json_path = "/dental_json/classification/calculus/calculus_qa_dialogue.jsonl"

# ====== 读取诊断 JSON ======
with open(input_json_path, "r", encoding="utf-8") as f:
    diagnosis_data = json.load(f)

# ====== 提取 <Answer> 部分 ======
for entry in diagnosis_data:
    cot_answer = entry.get("cot_answer", "")
    match = re.search(r"<Answer>(.*?)</Answer>", cot_answer, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot find <Answer> in entry id={entry.get('id')}")
    entry["answer_text"] = match.group(1).strip()

# ====== 只处理前 3 条 ======
diagnosis_data = diagnosis_data[:3000]

# ====== 生成多轮对话的 prompt 函数 ======
def generate_dialogue_prompt(diagnosis):
    first_answer = diagnosis.get("answer_text")
    question = diagnosis.get("question")
    if question is None:
        raise ValueError(f"Cannot find 'question' in entry id={diagnosis.get('id')}")

    prompt = f"""
You are an experienced oral healthcare specialist. Based on the following preliminary diagnosis, simulate a realistic multi-turn conversation between a patient and the doctor.

Diagnosis: {first_answer}  

Requirements:  
1. Generate at least 5 full dialogue turns (1 patient + 1 doctor = 1 turn).  
2. Patient questions should sound natural and realistic, reflecting key concerns relevant to the provided Diagnosis. The Diagnosis is based on a preliminary assessment from an oral image, so patient questions should realistically align with what a patient might ask a doctor in this context.  
3. Patient concerns should cover multiple aspects, such as:  
   - Understanding the condition (meaning, seriousness, commonness, progression).  
   - Treatment and management (options, medication, dentist visits, surgery, self-care).
   - Symptoms and prognosis (expected course, eating/speaking, recurrence, complications).  
   - Lifestyle and diet (food, drinks, smoking, alcohol, oral care routines).  
   - Prevention and oral care (future prevention, hygiene, dentist visits, products).  
   - Follow-up and long-term management (checkups, worsening, overall health).  
   - Impact on work/study (daily activities, time off).  
   - Treatment duration (time, visits, long/short term).  
   - Pain and comfort (pain relief, medication, sleep).  
   - Age-related concerns regarding recovery speed.  
4. Doctor answers should be professional, medically accurate, and easy for patients to understand.  
5. The doctor may suggest seeing a dentist, further tests, or referrals if appropriate.  
6. If a patient question involves an emergency or serious condition, the doctor should provide clear and direct guidance.  
7. The dialogue should end politely (e.g., with thanks or reassurance).  
8. Output format: JSON array only. Each turn should follow this format:  

[
  {{"Patient": "...", "Doctor": "..."}},
  {{"Patient": "...", "Doctor": "..."}}
]

Do not include any extra text outside the JSON array.  
"""
    return prompt

# ====== 支持断点续跑 ======
existing_ids = set()
if os.path.exists(output_json_path):
    with open(output_json_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                existing_ids.add(obj["id"])
            except:
                continue

print(f"已有 {len(existing_ids)} 条结果，将跳过这些条目。")

# ====== 逐条生成并写入 ======
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "a", encoding="utf-8") as f:  # 使用追加模式
    for entry in diagnosis_data:
        if entry["id"] in existing_ids:
            print(f"跳过已生成的 id={entry['id']}")
            continue

        prompt = generate_dialogue_prompt(entry)

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        dialogue_text = response.choices[0].message.content.strip()

        try:
            dialogue_json = json.loads(dialogue_text)
        except json.JSONDecodeError:
            print(f"解析失败 id={entry['id']}, 将跳过")
            continue

        out_obj = {
            "id": entry["id"],
            "image": entry.get("image"),
            "diagnosis": entry["answer_text"],
            "dialogue": dialogue_json
        }

        f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        f.flush()
        print(f"✅ 成功生成 id={entry['id']}")
