import os
import json
import base64
from PIL import Image
from io import BytesIO
import random

# 设置输入文件夹路径
input_folder = "/home/jinghao/projects/x-ray-VLM/R1/Sichuan_University_Dental_dataset/Pediatric dental disease detection dataset/Train/label"  # 替换为你的文件夹路径
output_file = "output.json"

Question_template = [
    "What abnormalities can be observed in this panoramic dental image?",
    "Are there any abnormalities in this dental panoramic X-ray?",
    "Can you identify any problems in this oral panoramic image?",
    "What issues are present in this panoramic view of the oral cavity?",
    "Do you notice any anomalies in this dental panoramic radiograph?",
    "Are there any visible concerns in this oral panoramic X-ray?",
    "What irregular findings are shown in this panoramic dental X-ray?",
    "Can you detect any abnormalities in this panoramic radiographic image?",
    "What unusual features stand out in this dental panoramic image?",
    "Are there any signs of pathology in this panoramic dental radiograph?",
]

def process_boxes(boxes, image_width, image_height):
    """
    处理 box 列表，根据中心点坐标合并并扩大 box。
    
    参数:
        boxes: list[list[int]]，每个元素代表一个 box，格式为 [x1, y1, x2, y2]
        image_width: int，图像宽度
        image_height: int，图像高度
    
    返回:
        list[list[int]]，处理后的 box 列表
    """
    
    # 扩大每个 box
    expanded_boxes = []
    for box in boxes:
        x1 = box[0] - 30
        y1 = box[1] - 30
        x2 = box[2] + 30
        y2 = box[3] + 30
        expanded_boxes.append([x1, y1, x2, y2])
    
    return expanded_boxes

# 函数：从 imageData 字段解码图像并获取宽高
def get_image_dimensions(image_data):
    try:
        # 解码 base64 图像数据
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        return image.width, image.height
    except Exception as e:
        print(f"无法解码图像: {e}")
        return 0, 0

# 函数：处理单个 JSON 文件，生成目标格式
def process_json_file(json_file, label_set):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图片名称和图像数据
    image_name = data.get("imagePath", "")
    image_data = data.get("imageData", "")
    
    # 获取图像宽高
    image_width, image_height = get_image_dimensions(image_data)

        # 获取标注信息
    shapes = data.get("shapes", [])
    answers = []
    precise_grounding_position = []
    category_list = []
    for shape in shapes:
        if shape["shape_type"] == "rectangle":
            label = shape["label"]
            if label == '其他':
                continue
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]
            precise_grounding_position.append([int(x1), int(y1), int(x2), int(y2)])

            box = f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"

            # 添加到 label 集合中
            label_set.add(label)

            # 构建 answer_sentence 和 precise_grounding_positions
            if label == '根尖周炎':
                label = "periapical periodontitis"
                # answer_sentence = f"The area <box>[{box}]</box> has {label}."
                answer_sentence = f"The {label} is found in the area <box>[{box}]</box>."
            elif label == '深窝沟':
                label = "deep pits and fissures"
                # answer_sentence = f"The area <box>[{box}]</box> has {label}."
                answer_sentence = f"The {label} is detected in the area <box>[{box}]</box>."
            elif label == '牙髓炎':
                label = "pulpitis"
                # answer_sentence = f"The area <box>[{box}]</box> has {label}."
                answer_sentence = f"The {label} is found in the area <box>[{box}]</box>."
            elif label == '牙齿发育异常':
                label = "Abnormal tooth development"
                answer_sentence = f"{label} is found in the area <box>[{box}]</box>."
            elif label == '龋病':
                label = "caries"
                # answer_sentence = f"The area <box>[{box}]</box> has {label}."
                answer_sentence = f"The {label} is observed in the area <box>[{box}]</box>."

            answers.append(answer_sentence)
            category_list.append(label)
    # 将所有 box 的信息拼接成一个句子
    answer_sentence = " ".join(answers)


    Contextual_bounding_boxes = process_boxes(precise_grounding_position, image_width, image_height)

    # 合并成一个字典
    result = {
        "image_name": "Sichuan_University_Dental_dataset/images/" + image_name,
        "image_width": image_width,
        "image_height": image_height,
        "source": "Sichuan University",
        "modality": "Panoramic X-ray",
        "Dentition Type": "Mixed",
        "Age Classification": "Child",
        "Question": random.choice(Question_template),
        "Annotations": answer_sentence,  # 一张图片的所有box信息都在这里
        "Precise Grounding Position": precise_grounding_position,
        "Contextual Grounding Position": Contextual_bounding_boxes,
        "Category": category_list
    }

    return result

# 主函数：遍历文件夹并生成合并 JSON
def process_folder(input_folder, output_file):
    all_results = []
    label_set = set()  # 用于统计 label 种类
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                all_results.append(process_json_file(json_file, label_set))
    # 保存为一个合并的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # 打印所有 label 种类
    print(f"发现的所有 label 种类: {sorted(label_set)}")
    print(f"处理完成，结果保存在 {output_file}")

# 执行脚本
process_folder(input_folder, output_file)