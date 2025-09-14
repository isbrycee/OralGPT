import os
import glob
import base64
import json
from openai import OpenAI

# ====== 配置 ======
client = OpenAI(
    api_key="sk-xx",  # 换成你的 key
    base_url="https://api.chatanywhere.org/v1"
)

root_dir = "./Oral-GPT-data/image/Vident-real"
splits = ["train", "val", "test"]

# ====== 输出 JSON 路径 ======
output_dir = "./Oral-GPT-data/dental_json/video"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "vident_captions.json")

# ====== 读取已生成的记录，实现断点续传 ======
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

# base64 编码函数
def encode_image_to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"url": f"data:image/png;base64,{b64}"}

# ====== 控制处理数量 ======
processed = 0
max_cases = 1000   # 👈 想处理几个 case，就改这里

# ====== 循环处理 ======
for split in splits:
    split_path = os.path.join(root_dir, split)
    case_dirs = [d for d in glob.glob(os.path.join(split_path, "*")) if os.path.isdir(d)]

    for case_dir in case_dirs:
        if processed >= max_cases:  # 达到上限就退出
            break

        gt_dir = os.path.join(case_dir, "GT")
        if not os.path.exists(gt_dir):
            continue

        # 相对路径，用于判断是否已处理
        rel_path = os.path.relpath(gt_dir, os.path.dirname(root_dir))
        if any(entry["image"] == rel_path for entry in results):
            print(f"跳过已处理: {rel_path}")
            continue

        # 获取所有帧
        frame_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        if not frame_files:
            continue

        # 抽帧
        selected_frames = frame_files[::80]

        # === 每个 case 单独构建 messages ===
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical dental expert. You are analyzing a real intra-oral surgical video recorded during conservative dental treatment. "
                    "The video captures complex clinical conditions inside the oral cavity, where the scene is crowded with multiple dental instruments and artifacts. "
                    "It may include conditions such as occlusions, frequent appearance variations, tool–tooth interactions, bleeding, water spray, splashing fluids, motion blur, strong light reflections, and occasional camera fouling. "
                    "The footage could show non-standard tools, intra-oral mirrors, and other interfering objects that partially obstruct the view."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Generate a detailed and structured description of the intra-oral surgical procedure shown in the video. "
                            "Focus on: (1) clinical environment and imaging conditions, (2) visible tools, tissues, and anatomy, (3) dynamic interactions during treatment, "
                            "(4) procedural sequence, (5) notable clinical findings and challenges, (6) possible procedural intents. "
                            "Conclude with a final summary that integrates descriptions of the above six aspects. "
                            "Do not add any instructions, suggestions, or optional tasks."
                        )
                    }
                ]
            }
        ]

        # 添加帧
        for frame in selected_frames:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": encode_image_to_data_uri(frame)
            })

        # 调用模型
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages
        )
        caption = response.choices[0].message.content

        # 保存 entry
        entry = {
            "id": f"video_caption_{len(results) + 1}",
            "image": rel_path,
            "caption": caption,
            "Modality": "Intraoral photograph",
            "category": "video"
        }
        results.append(entry)

        # ====== 每处理完一个 case 就立即写入 JSON ======
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        processed += 1  # 已处理 +1
        print(f"已处理 {processed}: {rel_path}")

    if processed >= max_cases:
        break

print(f"保存完成: {output_path}, 本次新增 {processed} 条记录, 总共 {len(results)} 条记录")
