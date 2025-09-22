import os
import json

def check_json_files(json_folder, image_folder):
    # 遍历文件夹下所有的 .json 文件
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder, filename)
            if "for_" not in filename:
                continue
            print(f"🔍 检查文件: {filename}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ 无法读取 {filename} : {e}")
                continue

            # 数据格式是一个二维嵌套数组，如示例中
            for block in data:
                for obj in block:
                    conversations = obj.get("conversations", [])
                    images = obj.get("images", [])

                    # 1. 统计 human 的 <image> 数量
                    human_image_count = 0
                    human_texts_with_images = []  # 保存包含 <image> 的 human 文本
                    for conv in conversations:
                        if conv["from"] == "human":
                            count = conv["value"].count("<image>")
                            if count > 0:
                                human_texts_with_images.append(conv["value"])
                            human_image_count += count

                    images_list_count = len(images)

                    if human_image_count != images_list_count:
                        if human_image_count > images_list_count:
                            diff = human_image_count - images_list_count
                            print(f"⚠️ {filename} : human 中 <image>（{human_image_count} 个）比 images 列表（{images_list_count} 个）多 {diff} 个")
                        else:
                            diff = images_list_count - human_image_count
                            print(f"⚠️ {filename} : images 列表（{images_list_count} 个）比 human 中 <image>（{human_image_count} 个）多 {diff} 个")
                            print(conversations)

                        # 打印有 <image> 的 human 文本内容
                        print(f"👉 相关 human 内容：")
                        for txt in human_texts_with_images:
                            print(f"   - {txt}")

                    # 2. 检查 gpt 的 value 中是否包含 <image>
                    for conv in conversations:
                        if conv["from"] == "gpt" and "<image>" in conv["value"]:
                            print(f"❌ {filename} : gpt 的回复中包含 <image> -> {conv['value']}")

                    # 3. 检查 image list 中的图片是否存在
                    for img_path in images:
                        img_name = os.path.basename(img_path)
                        img_full_path = os.path.join(image_folder, img_name)
                        if not os.path.exists(img_full_path):
                            print(f"❌ {filename} : 缺失图片文件 {img_name} (在 {image_folder} 找不到)")

if __name__ == "__main__":
    json_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/en_image_pair"   # 👉 JSON 文件夹路径
    image_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/en_image_pair/images" # 👉 图像文件夹路径
    check_json_files(json_folder, image_folder)
