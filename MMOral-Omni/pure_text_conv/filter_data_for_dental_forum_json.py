import json

def check_and_fix_json(json_data):
    fixed_data = []
    for idx, item in enumerate(json_data):
        if "conversations" not in item or not isinstance(item["conversations"], list):
            print(f"[Warning] item[{idx}] missing or invalid 'conversations', skipped.")
            continue

        convs = item["conversations"]
        fixed_convs = []
        prev = None
        modifications = []

        for i, conv in enumerate(convs):
            # 必须有 "from" 和 "value"
            if "from" not in conv or "value" not in conv:
                modifications.append(
                    f"item[{idx}] conv[{i}] missing 'from' or 'value' -> removed")
                continue

            sender = conv["from"]
            if sender not in ("human", "gpt"):
                modifications.append(
                    f"item[{idx}] conv[{i}] invalid from={sender} -> removed")
                continue

            # 检查正确顺序：必须 human→gpt→human...
            if prev is None:
                # 第一个必须是 human
                if sender != "human":
                    modifications.append(
                        f"item[{idx}] conv[{i}] first message not human (from={sender}) -> removed")
                    continue
            else:
                # 检查交替
                if prev == sender:
                    modifications.append(
                        f"item[{idx}] conv[{i}] invalid order (two consecutive '{sender}') -> removed")
                    continue

            fixed_convs.append(conv)
            prev = sender

        # 如果最后是 human，把它删掉
        if fixed_convs and fixed_convs[-1]["from"] == "human":
            modifications.append(f"item[{idx}] last conversation is human -> removed")
            fixed_convs.pop()

        # 检查是否至少保留了完整的一对 human-gpt
        valid_pairs = 0
        for j in range(0, len(fixed_convs) - 1, 2):
            if fixed_convs[j]["from"] == "human" and fixed_convs[j+1]["from"] == "gpt":
                valid_pairs += 1

        if valid_pairs == 0:
            print(f"item[{idx}] deleted because it does not contain at least one human-gpt pair.")
            continue

        if modifications:
            print("\n".join(modifications))
            print(f"--- After fixing item[{idx}], kept {valid_pairs} valid pairs ---")

        item["conversations"] = fixed_convs
        fixed_data.append(item)

    return fixed_data



if __name__ == "__main__":
    # 假设 JSON 文件叫 input.json
    with open("/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/pure_text_dental_QA_from_dental_forum_shareGPT_for_sft.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_data = check_and_fix_json(data)

    # 保存修复后的结果
    with open("/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/pure_text_dental_QA_from_dental_forum_shareGPT_for_sft_checked.json", "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    print("检查与修复完成，结果已保存到 fixed_output.json")
