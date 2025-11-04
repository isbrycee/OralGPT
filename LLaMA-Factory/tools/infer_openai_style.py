
import os
import json
import base64
from openai import OpenAI

def encode_image(image_path):
    """将图片编码为 base64。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def process_test_json(
    input_json: str,
    output_json: str,
    image_folder: str,
):
    client = OpenAI(
        base_url="http://147.8.222.102:8000/v1",  # 替代 api_base
        api_key="0"                       # llamafactory 默认不需要真实 key
    )
    with open(input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = []

    for idx, sample in enumerate(samples):
        system_prompt = sample.get("system", "")
        messages_in = sample.get("messages", [])
        images = sample.get("images", [])

        # 找到最后一个 user role 的消息作为问题
        user_messages = [m for m in messages_in if m.get("role") == "user"]
        if not user_messages:
            print(f"Sample {idx} 没有 user 消息，跳过。")
            continue
        question = user_messages[-1]["content"].replace("<image>", "")

        # 组装 messages
        user_content = [{"type": "text", "text": question}]
        for img_path in images:
            img_path = os.path.join(image_folder, img_path)
            
            if not os.path.exists(img_path):
                print(f"警告: 图片 {img_path} 不存在，跳过。")
                continue
            image_b64 = encode_image(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 调用模型
        try:
            response = client.chat.completions.create(
                messages=messages,
                model="test"
            )
            answer = response.choices[0].message.content
            # print(answer)
        except Exception as e:
            answer = f"Error: {e}"

        result_item = {
            "index": idx,
            "system": system_prompt,
            "question": question,
            "images": images,
            "answer": answer
        }
        results.append(result_item)

        print(f"Processed sample {idx} ✅")

    # 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"所有结果已保存到 {output_json}")

if __name__ == "__main__":
    image_folder = "/home/jinghao/projects/positioning-error/" 
    input_json = "/home/jinghao/projects/positioning-error/OPG_Positioning_Error_train.json"
    output_json = "/home/jinghao/projects/positioning-error/train_infer_results.json"
    process_test_json(input_json, output_json, image_folder)