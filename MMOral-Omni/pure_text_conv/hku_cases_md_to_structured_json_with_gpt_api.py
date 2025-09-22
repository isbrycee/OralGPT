import os
import re
import json
import openai
from openai import OpenAI
import ast
import time
from tqdm import tqdm
from typing import Tuple, List, Any

# 设置OpenAI API密钥
client = OpenAI(
    api_key="sk-",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)

SYSTEM_PROMPT_endodontics = """
You will be given a **Markdown-formatted oral clinical case presentation**, containing text and images mixed with section headings.

Your task is to generate **two structured outputs** based on the case:

1. **A simulated patient–chatbox conversation**
2. **A set of oral knowledge Q&A pairs**

---

## Part 1: Simulated Patient–Chatbox Conversation

### Conversation Flow (strictly follow section mapping)

1. **Patient (Opening)**
    
    Uses:
    
    - `# Chief Complaint`
    - `# Medical History`
    - `# Dental History`
    Ends with: *“Please help me diagnose my oral condition.”*
2. **Chatbox (Requests more info)**
    
    Requests EOE and IOE.
    
3. **Patient (Provides info)**
    
    Uses:
    
    - `# Extra-oral Examination (EOE)`
    - `# Intra-oral Examination (IOE)`
4. **Chatbox (Requests images)**
5. **Patient (Uploads radiographic image)**
    - Replace case images `![](path)` → `<image>`
    - If the **Radiographic Findings** section refers to multiple figures (e.g., “see Fig. 1 and Fig. 2”), the patient must include the same number of `<image>` entries in sequence (e.g., `<image><image>`).
    - Patient then asks: *“From this radiographic image, what findings can you observe?”*
6. **Chatbox (Provides radiographic findings)**
    
    Uses:
    
    - `# Radiographic Findings`
7. **Patient (Requests diagnosis & treatment plan)**
8. **Chatbox (Provides diagnosis & options)**
    
    Uses:
    
    - `# Pretreatment Diagnosis`
    - `# Treatment Plan`
    - `# Recommended`
    - `# Alternative`
    - `# Restorative`
9. **Patient (Requests summary)**
10. **Chatbox (Provides summary)**
    
    Uses: `# Clinical Procedures: Treatment Record`
    
    *Ignore images in this section.*
    
11. **Patient (Requests post-treatment guidance)**
12. **Chatbox (Provides post-treatment evaluation)**
    
    Uses: `# Post-Treatment Evaluation`
    

---

### Image Processing Rules

- Replace all `![](image_path)` with `<image>`.
- Each `<image>` corresponds to the nearest subsequent **Figure xxx** number.
- If **Radiographic Findings** refers to multiple figures, include the exact same number of `<image>` tags for that dialogue turn.
- Collect image paths into a separate list, preserving the same order as they appear in the conversation.

---

### ✅ Output Format for Part 1

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

---

## Part 2: Oral Knowledge Q&A Pairs

- Extract **questions** from:
    - `# Self Study Questions`
- Extract **answers** from:
    - `# Answers to Self-Study Questions`
- Each Q&A is generated as a pair.

---

### ✅ Output Format for Part 2

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

---

## Important Notes

1. Ignore all images inside **Clinical Procedures: Treatment Record** section.
2. Respect section-to-turn mapping when creating dialogue.
3. Ensure `<image>` order in dialogue and “images” list matches exactly the input document order.
4. If **Radiographic Findings** mentions multiple figures, patient’s dialogue must have the same number of `<image>` tags.
5. "```json" and "```" markers must be included in the final output to denote JSON blocks.

---

✅ **Your output must contain both parts**:

1. Patient–chatbox conversations + ordered image list (Part 1).
2. Knowledge Q&A pairs (Part 2).

"""
SYSTEM_PROMPT_implant_dentistry = """
You will be given a **Markdown-formatted oral clinical case presentation**, containing text and images mixed with section headings.

Your task is to generate **two structured outputs** based on the case:

1. **A simulated patient–chatbox conversation**
2. **A set of oral knowledge Q&A pairs**

---

## Part 1: Simulated Patient–Chatbox Conversation

### Conversation Flow (strictly follow section mapping)

1. **Patient (Opening)**
    
    Uses:
    
    - `# CASE STORY`
    - `# Medical History`
    - `# Social History`
    Ends with: *“Please help me diagnose my oral condition.”*
2. **Chatbox (Requests more info)**
    
    Requests EOE, IOE, and Occlusion.
    
3. **Patient (Provides info)**
    
    Uses:
    
    - `# Extraoral Examination (EOE)`
    - `# Intraoral Examination (IOE)`
    - `# Occlusion`
4. **Chatbox (Requests images)**
5. **Patient (Uploads radiographic image)**
    - Replace case images `![](path)` → `<image>`
    - If the **Radiographic Examination** section refers to multiple figures (e.g., “see Fig. 1 and Fig. 2”), the patient must include the same number of `<image>` entries in sequence (e.g., `<image><image>`).
    - Patient then asks: *“From this radiographic image, what findings can you observe?”*
6. **Chatbox (Provides radiographic findings)**
    
    Uses:
    
    - `# # Radiographic Examination`
7. **Patient (Requests diagnosis & treatment plan)**
8. **Chatbox (Provides diagnosis & options)**
    
    Uses:
    
    - `# Diagnosis`
    - `# Treatment Plan`
    - `# Recommended`
    - `# Alternative`
    - `# Restorative`
9. **Patient (Requests summary)**
10. **Chatbox (Provides summary)**
    
    Uses: 
    - `# Clinical Procedures: Treatment Record`
    - `# Discussion`
    
    *Ignore images in this section.*
    
11. **Patient (Requests post-treatment guidance)**
12. **Chatbox (Provides post-treatment evaluation)**
    
    Uses: `# Post-Treatment Evaluation`
    

---

### Image Processing Rules

- Replace all `![](image_path)` with `<image>`.
- Each `<image>` corresponds to the nearest subsequent **Figure xxx** number.
- If **Radiographic Findings** refers to multiple figures, include the exact same number of `<image>` tags for that dialogue turn.
- Collect image paths into a separate list, preserving the same order as they appear in the conversation.

---

### ✅ Output Format for Part 1

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

---

## Part 2: Oral Knowledge Q&A Pairs

- Extract **questions** from:
    - `# Self Study Questions`
- Extract **answers** from:
    - `# Answers to Self-Study Questions`
- Each Q&A is generated as a pair.

---

### ✅ Output Format for Part 2

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

---

## Important Notes

1. Ignore all images inside **Clinical Procedures: Treatment Record** section.
2. Respect section-to-turn mapping when creating dialogue.
3. Ensure `<image>` order in dialogue and “images” list matches exactly the input document order.
4. If **Radiographic Findings** mentions multiple figures, patient’s dialogue must have the same number of `<image>` tags.
5. "```json" and "```" markers must be included in the final output to denote JSON blocks.

---

✅ **Your output must contain both parts**:

1. Patient–chatbox conversations + ordered image list (Part 1).
2. Knowledge Q&A pairs (Part 2).

"""
SYSTEM_PROMPT_Periodontics = """
You will be given a **Markdown-formatted oral clinical case presentation**, containing text and images mixed with section headings.

Your task is to generate **two structured outputs** based on the case:

1. **A simulated patient–chatbox conversation**
2. **A set of oral knowledge Q&A pairs**

---

## Part 1: Simulated Patient–Chatbox Conversation

### Conversation Flow (strictly follow section mapping)

1. **Patient (Opening)**
    
    Uses:
    
    - `# CASE STORY`
    - `# Medical History or # Dental History`
    - `# Social History`
    Ends with: *“Please help me diagnose my oral condition.”*
    If these utilized section refers to multiple figures (e.g., “see Fig. 1 and Fig. 2”), the patient must include the same number of `<image>` entries in sequence (e.g., `<image><image>`).
2. **Chatbox (Requests more info)**
    
    Requests EOE, IOE, Occlusion, or other Examination
    
3. **Patient (Provides info)**
    
    Uses:
    
    - `# Extraoral Examination (EOE)`
    - `# Intraoral Examination (IOE)`
    - `# Occlusion`
    - `# Other Examination`
4. **Chatbox (Requests images)**
5. **Patient (Uploads radiographic image)**
    - Replace case images `![](path)` → `<image>`
    - If the **Radiographic Examination** section refers to multiple figures (e.g., “see Fig. 1 and Fig. 2”), the patient must include the same number of `<image>` entries in sequence (e.g., `<image><image>`).
    - Patient then asks: *“From this radiographic image, what findings can you observe?”*
6. **Chatbox (Provides radiographic findings)**
    
    Uses:
    
    - `# Radiographic Examination`
7. **Patient (Requests diagnosis & treatment plan)**
8. **Chatbox (Provides diagnosis & options)**
    
    Uses:
    
    - `# Diagnosis`
    - `# Treatment Plan`
    - `# Recommended`
    - `# Alternative`
    - `# Restorative`
9. **Patient (Requests discussion about the case)**
10. **Chatbox (Provides discussion about the case)**
    
    Uses: 
    - `# Discussion`
    *Ignore images in this section.*
    
---

### Image Processing Rules

- Replace all `![](image_path)` with `<image>`.
- Each `<image>` corresponds to the nearest subsequent **Figure xxx** number.
- If **Radiographic Findings** refers to multiple figures, include the exact same number of `<image>` tags for that dialogue turn.
- Collect image paths into a separate list, preserving the same order as they appear in the conversation.

---

### ✅ Output Format for Part 1

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

---

## Part 2: Oral Knowledge Q&A Pairs

- Extract **questions** from:
    - `# Self-Study Questions`
- Extract **answers** from:
    - `# TAKE-HOME POINTS`
- Each Q&A is generated as a pair.
- *Ignore images in `# TAKE-HOME POINTS` section.*
---

### ✅ Output Format for Part 2

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

---

## Important Notes

1. Ignore all images inside **Discussion** section.
2. Respect section-to-turn mapping when creating dialogue.
3. Ensure `<image>` order in dialogue and “images” list matches exactly the input document order.
4. If **CASE STORY** or **Radiographic Findings** mentions multiple figures, patient’s dialogue must have the same number of `<image>` tags.
5. "```json" and "```" markers must be included in the final output to denote JSON blocks.

---

✅ **Your output must contain both parts**:

1. Patient–chatbox conversations + ordered image list (Part 1).
2. Knowledge Q&A pairs (Part 2).

"""
SYSTEM_PROMPT_Pediatric_Dentistry = """
You will be given a **Markdown-formatted oral clinical case presentation**, containing text and images mixed with section headings.

Your task is to generate **two structured outputs** based on the case:

1. **A simulated patient–chatbox conversation**
2. **A set of oral knowledge Q&A pairs**

---

## Part 1: Simulated Patient–Chatbox Conversation

The case will include information across multiple sections such as (but not limited to):

# Presenting Patient
# Chief Complaint
# Medical History or # Dental History
# Social History
# Extraoral Examination (EOE)
# Intraoral Examination (IOE)
# FUNDAMENTAL POINT
# Diagnostic Tools
# Differential Diagnosis Developmental
# Diagnosis and Problem List
# Comprehensive Treatment Plan
# Prognosis and Discussion
# Common Complications and Alternative Treatment Plans
# BACKGROUND INFORMATION
Using the details from these sections, construct a simulated dialogue between a pediatric patient (as “human”) and a chatbox (as “gpt”) about oral disease/care.

---

### Image Processing Rules

- Replace all `![](image_path)` with `<image>`.
- Each `<image>` corresponds to the nearest subsequent **Figure xxx** number.
- If multiple figures are mentioned in one turn, include the exact same number of `<image>` tags for that dialogue turn.
- Collect image paths into a separate list, preserving the same order as they appear in the conversation.

---

### ✅ Output Format for Part 1

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

---

## Part 2: Oral Knowledge Q&A Pairs

- Extract **questions** from:
    - `# Self-Study Questions`
- Extract **answers** from:
    - `# SELF-STUDY ANSWERS`
- Each Q&A is generated as a pair.
- *Ignore images in `# SELF-STUDY ANSWERS` section.*
---

### ✅ Output Format for Part 2

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

---

## Important Notes

1. Completely **ignore images** in the **Discussion** section.
2. Ensure the conversation follows the patient–first pattern, aligned to the case sections.
3. `<image>` tags must appear in the **correct order**, matching exactly the `"images"` list.
4. **`<image>` may only appear in the patient’s turns** and never in the chatbox’s turns.
5. When multiple figures are mentioned in one turn, the number of `<image>` tags must exactly equal the number of figures
5. "```json" and "```" markers must be included in the final output to denote JSON blocks.

---

✅ **Your output must contain both parts**:

1. Patient–chatbox conversations + ordered image list (Part 1).
2. Knowledge Q&A pairs (Part 2).

"""
SYSTEM_PROMPT_Restorative_and_Reconstructive_Dentistry = """
You will be given a **Markdown-formatted oral clinical case presentation**, containing text and images mixed with section headings.

Your task is to generate **one structured outputs** based on the case:**A simulated patient–chatbox conversation**

---

## Simulated Patient–Chatbox Conversation

The case will include information across multiple sections such as (but not limited to):

# SUMMARY OF EXAMINATION AND DIAGNOSIS
# Dentition
# Periodontium
# TMJs
# Muscles
# Occlusion
# Aesthetics
# THE 10 DECISIONS
# SUMMARY OF TREATMENT PLAN
# SUMMARY OF TREATMENT SEQUENCE

Using the details from these sections, construct a simulated dialogue between a pediatric patient (as “human”) and a chatbox (as “gpt”) about oral disease/care.

---

### Image Processing Rules

- Replace all `![](image_path)` with `<image>`.
- Each `<image>` corresponds to the nearest subsequent **Figure xxx** number.
- If multiple figures are mentioned in one turn, include the exact same number of `<image>` tags for that dialogue turn.
- Collect image paths into a separate list, preserving the same order as they appear in the conversation.

### Important Notes

1. Completely **ignore images** in the **SUMMARY OF TREATMENT SEQUENCE** section.
2. Ensure the conversation follows the patient–first pattern, aligned to the case sections.
3. `<image>` tags must appear in the **correct order**, matching exactly the `"images"` list.
4. **`<image>` may only appear in the patient’s turns** and never in the chatbox’s turns.
5. When multiple figures are mentioned in one turn, the number of `<image>` tags must exactly equal the number of figures
5. "```json" and "```" markers must be included in the final output to denote JSON blocks.

---

### ✅ Output Format for Part 1

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

"""
SYSTEM_PROMPT_Orthodontics = """
You will be given a **Markdown-formatted oral clinical case presentation**, containing text and images mixed with section headings.

Your task is to generate **one structured outputs** based on the case:**A simulated patient–chatbox conversation**

---

## Simulated Patient–Chatbox Conversation

The case will include information across multiple sections such as (but not limited to):

# Extra-oral
# Intra-oral
# Summary
# Treatment Progress
# Treatment Plan
# Several Questions and Answers

Using the details from these sections, construct a simulated dialogue between a pediatric patient (as “human”) and a chatbox (as “gpt”) about oral disease/care.

---

### Image Processing Rules

- Replace all `![](image_path)` with `<image>`.
- Each `<image>` corresponds to the nearest subsequent **Figure xxx** number.
- If multiple figures are mentioned in one turn, include the exact same number of `<image>` tags for that dialogue turn.
- Collect image paths into a separate list, preserving the same order as they appear in the conversation.

### Important Notes

1. Completely **ignore images** in the **SUMMARY OF TREATMENT SEQUENCE** section.
2. Ensure the conversation follows the patient–first pattern, aligned to the case sections.
3. `<image>` tags must appear in the **correct order**, matching exactly the `"images"` list.
4. **`<image>` may only appear in the patient’s turns** and never in the chatbox’s turns.
5. When multiple figures are mentioned in one turn, the number of `<image>` tags must exactly equal the number of figures
5. "```json" and "```" markers must be included in the final output to denote JSON blocks.

---

### ✅ Output Format for Part 1

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

"""


def extract_lists_from_response(text: str) -> Tuple[bool, List[Any]]:
    """
    从输入字符串中解析出两个被 ```json ... ``` 包围的 JSON 列表。
    
    返回:
        (flag, result)
        - flag: True 表示成功且刚好两个 list，False 表示出错
        - result: 成功时是两个解析后的 list，失败时是空列表
    """
    # 找出所有被 ```json ... ``` 包裹的内容
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)

    # 必须正好有两个
    if len(blocks) != 2:
        return False, [], []

    json_lists = []
    for block in blocks:
        try:
            parsed = json.loads(block.strip())
            if not isinstance(parsed, list):
                return False, [], []  # 如果不是 list，就报错
            json_lists.append(parsed)
        except json.JSONDecodeError:
            return False, [], []

    return True, json_lists[0], json_lists[1]

def extract_lists_from_response_for_only_case_study_wo_QA(text: str) -> Tuple[bool, List[Any]]:
    """
    从输入字符串中解析出两个被 ```json ... ``` 包围的 JSON 列表。
    
    返回:
        (flag, result)
        - flag: True 表示成功且刚好两个 list，False 表示出错
        - result: 成功时是两个解析后的 list，失败时是空列表
    """
    # 找出所有被 ```json ... ``` 包裹的内容
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)

    # 必须正好有两个
    if len(blocks) != 1:
        return False, [], []

    json_lists = []
    for block in blocks:
        try:
            parsed = json.loads(block.strip())
            if not isinstance(parsed, list):
                return False, [], []  # 如果不是 list，就报错
            json_lists.append(parsed)
        except json.JSONDecodeError:
            return False, [], []

    return True, json_lists[0], []


def extract_sections_from_md_endodontics(md_text):
    """
    根据 # Case I/II/III: 和 # Reference 分割
    返回 [(case_title, case_content), ...]
    """
    # 支持罗马数字的正则，匹配一个 Case 段落到下一个 Case 或文件结束
    case_pattern = r"(Case\s+[IVXLCDM]+:.*?)(?=Case\s+[IVXLCDM]+:|$)"
    matches = re.findall(case_pattern, md_text, re.S)

    sections = []
    for match in matches:
        lines = match.strip().splitlines()
        case_title = lines[0].strip()

        # 按 # Reference 分割
        ref_split = re.split(r"(?m)^# Reference", match, maxsplit=1)
        if len(ref_split) > 1:
            case_content = ref_split[0].strip()   # Reference 之前的部分
            # reference_content = ref_split[1].strip()  # 如果你之后需要 Reference 部分，可以用这个变量
        else:
            case_content = match.strip()

        # sections.append((case_title, case_content))
        sections.append(case_content)
    return sections

def extract_sections_from_md_implant_dentistry_or_Periodontics(md_text):
    """
    根据 "# CASE STORY" 分割
    返回 [case_content1, case_content2, ...]
    """
    # 匹配 "# CASE STORY" 开头到下一个 "# CASE STORY" 或结尾
    case_pattern = r"(# CASE STORY.*?)(?=# CASE STORY|$)"
    matches = re.findall(case_pattern, md_text, re.S)

    sections = [m.strip() for m in matches]
    return sections

def extract_sections_from_md_implant_dentistry_or_Pediatric_Dentistry(md_text):
    """
    匹配 "# A. Presenting Patient" 开始，
    到 "# SELF-STUDY ANSWERS" 之后的第一个 '#' 标题之前的部分。
    返回 [section1, section2, ...]
    """
    # 正则分三步：
    # 1. 定位 "# A. Presenting Patient" 开头
    # 2. 一直到 "# SELF-STUDY ANSWERS" 及其后内容
    # 3. 在遇到下一个 '#' 开头的标题时停止（但不会匹配 # SELF-STUDY ANSWERS 自己）
    pattern = r"(# A\. Presenting Patient.*?# SELF-STUDY ANSWERS.*?)(?=\n#(?! SELF-STUDY ANSWERS)|$)"

    matches = re.findall(pattern, md_text, re.S)

    sections = [m.strip() for m in matches]
    return sections

def extract_sections_from_md_Restorative_and_Reconstructive_Dentistry(md_text):
    """
    提取片段：
    每个片段 = 从 SUMMARY 之前最近的 # 开始，一直到下一个 SUMMARY 之前为止（包含 SUMMARY 本身）。
    返回 [section1, section2, ...]
    """
    sections = []
    summaries = list(re.finditer(r"\n# SUMMARY OF EXAMINATION AND DIAGNOSIS", md_text))
    n = len(summaries)

    for i, match in enumerate(summaries):
        summary_start = match.start()

        # 找到此 SUMMARY 之前最近的 #
        prev_hash_match = list(re.finditer(r"\n#", md_text[:summary_start]))
        if not prev_hash_match:
            continue  # 如果找不到，则跳过
        start = prev_hash_match[-1].start()

        # 结束位置 = 下一个 SUMMARY 的开始 或 文末
        end = summaries[i+1].start() if i+1 < n else len(md_text)

        section = md_text[start:end].strip()
        if section:
            sections.append(section)

    return sections

def extract_sections_from_md_Orthodontics(md_text):
    """
    根据 '# CASE' 或 '# Case 3.6' 等形式进行分割。
    每个 '# CASE...' 到下一个 '# CASE...' 之间的内容作为一个 section。
    返回 [section1, section2, ...]
    """
    pattern = r"(#\s*CASE[^\n]*.*?)(?=\n#\s*CASE|\Z)"

    # ignore case, allow . and space etc.
    matches = re.findall(pattern, md_text, re.S | re.I)

    sections = [m.strip() for m in matches]
    return sections


def process_md_file(file_path, file_index, total_files):
    """处理单个MD文件并返回解析结果"""
    filename = os.path.basename(file_path)
    print(f"\n📄 处理文件 ({file_index}/{total_files}): {filename}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割Case和Reference部分
    cases = extract_sections_from_md_Orthodontics(content)

    print(f"  发现 {len(cases)} 个案例")
    
    all_list1 = []
    all_list2 = []
    
    # 使用tqdm创建进度条
    for i, case in enumerate(tqdm(cases, desc="  处理案例", unit="case")):
        success_flag = False
        retries = 0
        max_retries = 10

        while not success_flag and retries < max_retries:
            try:
                retries += 1
                start_time = time.time()
                response = client.chat.completions.create(
                    model="glm-4.5", #  glm-4.5 GLM-4.5-Flash
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_Orthodontics},
                        {"role": "user", "content": case}
                    ],
                    temperature=0.1,
                )
                api_time = time.time() - start_time

                # 提取并解析响应中的列表
                response_content = response.choices[0].message.content
                print("response: \n")
                print(response_content)  # 调试输出完整响应内容
                success_flag, conv_cases, conv_qa_paris = extract_lists_from_response_for_only_case_study_wo_QA(response_content)

                if success_flag:
                    all_list1.append(conv_cases)
                    all_list2.append(conv_qa_paris)
                    tqdm.write(f"    ✅ 案例 {i+1}/{len(cases)} 成功 (API耗时: {api_time:.2f}s, 尝试: {retries})")
                else:
                    tqdm.write(f"    ⚠️ 案例 {i+1}/{len(cases)} 第 {retries} 次尝试失败: 未找到两个有效列表")
            
            except Exception as e:
                tqdm.write(f"    ❌ 案例 {i+1}/{len(cases)} 第 {retries} 次尝试错误: {str(e)}")
        
        # 如果多次尝试都失败
        if not success_flag:
            tqdm.write(f"    ❌ 案例 {i+1}/{len(cases)} 最终失败 (已尝试 {max_retries} 次)")

    return all_list1, all_list2

def process_folder(folder_path):
    """处理整个文件夹中的MD文件"""
    # 获取所有MD文件
    md_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    total_files = len(md_files)
    
    if total_files == 0:
        print("⚠️ 错误: 未找到任何.md文件")
        return
    
    print(f"🔍 在文件夹中发现 {total_files} 个.md文件")
    
    master_list1 = []
    master_list2 = []
    processed_files = 0
    
    # 使用tqdm创建文件处理进度条
    for i, filename in enumerate(tqdm(md_files, desc="处理文件", unit="file")):
        file_path = os.path.join(folder_path, filename)
        if 'Orthodontics' not in filename:
            continue
        list1, list2 = process_md_file(file_path, i+1, total_files)
        master_list1.extend(list1)
        master_list2.extend(list2)
        processed_files += 1
    
    # 保存结果
    with open('hku_textbooks_Orthodontics_case_studies_conv.json', 'w', encoding='utf-8') as f:
        json.dump(master_list1, f, ensure_ascii=False, indent=2)
    
    with open('hku_textbooks_Orthodontics_QA_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(master_list2, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 处理完成!")
    print(f"  共处理 {processed_files} 个文件")
    print(f"  共解析 {len(master_list1)} 个列表项")
    print(f"  结果已保存到 output_list1.json 和 output_list2.json")

if __name__ == "__main__":
    folder_path = '/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/en_ocr_md'
    process_folder(folder_path)
