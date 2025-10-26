import json
import re

def fix_image_tags(input_path: str, output_path: str):
    # 读取 JSON 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_fixed = 0  # 统计修正的样本数量

    # 遍历每个样本
    for i, item in enumerate(data):
        images = item.get("images", [])
        conversations = item.get("conversations", [])
        human_msg = None

        # 找到 human 的消息
        for conv in conversations:
            if conv.get("from") == "human":
                human_msg = conv
                break

        if not human_msg:
            print(f"第 {i} 个样本没有 human 消息，跳过")
            continue  # 没有 human 消息，跳过

        value = human_msg.get("value", "")
        existing_count = len(re.findall(r"<image>", value))
        expected_count = len(images)

        if len(images) > 1:
            continue
        if len(images) == 1 and '7.1' in images[0]:
            continue
        # 检查并修正
        if existing_count < expected_count:
            missing = expected_count - existing_count
            human_msg["value"] = "<image>" * missing + value
            total_fixed += 1
            print(f"第 {i} 个样本修正：补充了 {missing} 个 <image> 标签")

    # 保存修正后的结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n共修正了 {total_fixed} 个样本，已保存到: {output_path}")

if __name__ == "__main__":
    input_json = "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/0.1_sft_multimodal_ALL_shareGPT_woThink_final.json"   # 输入文件路径
    output_json = "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/0.1_sft_multimodal_ALL_shareGPT_woThink_final_checked.json"  # 输出文件路径
    fix_image_tags(input_json, output_json)
    print(f"修正完成，已保存到: {output_json}")
