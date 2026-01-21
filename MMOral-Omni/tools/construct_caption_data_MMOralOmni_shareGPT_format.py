import json
import random
import re

# 你提供的人类提示替换列表
human_prompts = [
    "Please generate a caption for this image.",
    "Please describe this image comprehensively.",
    "Generate a caption for this image and describe it in detail.",
    "Please provide a detailed description for the uploaded image.",
    "Create a comprehensive image description.",
    "Write a caption for this image and thoroughly describe what’s shown."
    "Please write a complete, detailed description of this image."
    "Give me a full description of this image."
    "Produce a detailed description for this picture."
    "Provide an in‑depth visual description of the image."
]

# 读取原始 JSON
with open("/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/0.1_sft_multimodal_ALL_shareGPT.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = []

for item in data:
    # 仅保留 images 长度为 1 的条目
    if not isinstance(item.get("images"), list) or len(item["images"]) != 1:
        continue

    conv = item.get("conversations", [])
    if len(conv) != 2:
        # print(conv)
        continue

    # 找到 gpt 的回复
    gpt_value = conv[1].get("value", "")

    # 提取 <Caption>...</Caption>
    match = re.search(r"<[Cc]aption>(.*?)</[Cc]aption>", gpt_value, re.S)
    if not match:
        # print("no caption found!!")
        if "cephalometric radiograph" in gpt_value:
            continue
        if "<caption>" not in gpt_value:
            caption_text = gpt_value
            # print(gpt_value)
    else:
        caption_text = match.group(1).strip()

    # 修改 gpt 内容
    conv[1]["value"] = caption_text

    # 修改 human 内容（随机选一个）
    conv[0]["value"] = "<image> " + random.choice(human_prompts)

    output.append(item)


with open("/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/9.1_sft_panoramicImage_ReportVQAChat_MMOralOPG_shareGPT.json", "r", encoding="utf-8") as f1:
    data_OPG = json.load(f1)

for item in data_OPG:
    # 仅保留 images 长度为 1 的条目
    if not isinstance(item.get("images"), list) or len(item["images"]) != 1:
        continue

    # 直接在 item 上截取前两个，保证一定成功
    item["conversations"] = item["conversations"][:2]

    # 修改 human 内容
    item["conversations"][0]["value"] = "<image> " + random.choice(human_prompts)

    output.append(item)


# 保存处理后的 JSON
with open("/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/0.3_sft_multimodal_ALL_shareGPT_only_for_caption_generation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(len(output))
print("处理完成，已输出到 output.json")
