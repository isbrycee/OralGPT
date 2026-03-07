import json
import os
import random
from PIL import Image
import json

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


def process_coco_json(input_file):
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取类别信息
    categories_1 = {cat['id']: cat['name'] for cat in data['categories_1']}
    categories_2 = {cat['id']: cat['name'] for cat in data['categories_2']}
    categories_3 = {cat['id']: cat['name'] for cat in data['categories_3']}
    
    # 用于存储结果的列表
    results = []

    # 遍历每张图片
    images = {img['id']: img for img in data['images']}
    annotations = data.get('annotations', [])
    
    # 按图片分组
    image_to_boxes = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_to_boxes:
            image_to_boxes[image_id] = []
        image_to_boxes[image_id].append(ann)
    
    for image_id, boxes in image_to_boxes.items():
        # 获取图片信息
        image_info = images.get(image_id, {})
        image_name = image_info.get('file_name', '')
        image_width = image_info.get('width', 0)
        image_height = image_info.get('height', 0)
        
        # 构建 Answer 句子
        answers = []
        category_list = []
        for box in boxes:
            tooth_id = f"{categories_1.get(box['category_id_1'], '')}{categories_2.get(box['category_id_2'], '')}"
            disease_name = categories_3.get(box['category_id_3'], '')
            if disease_name == "Deep Caries" or disease_name == "Caries":
                disease_name = "caries"
                answer_sentence = f"The {disease_name} is found in tooth {tooth_id}."
                # answers.append(f"Tooth #{tooth_id} has {disease_name}.")
            elif disease_name == "Impacted":
                disease_name = "impacted"
                # answers.append(f"Tooth {tooth_id} is {disease_name}.")
                answer_sentence = f"Tooth {tooth_id} is {disease_name}."
            elif disease_name == "Periapical Lesion":
                disease_name = "periapical lesion"
                # answers.append(f"Tooth #{tooth_id} has {disease_name}.")
                answer_sentence = f"The {disease_name} is found in tooth {tooth_id}."
            answers.append(answer_sentence)
            category_list.append(disease_name)
        # 将所有 box 的信息拼接成一个句子
        answer_sentence = " ".join(answers)

        # 获取每张图片的精准定位信息
        precise_grounding_positions = [
            [int(box['bbox'][0]),  # x1
            int(box['bbox'][1]),  # y1
            int(box['bbox'][0] + box['bbox'][2]),  # x2 = x1 + w
            int(box['bbox'][1] + box['bbox'][3])]  # y2 = y1 + h
            for box in boxes if 'bbox' in box and len(box['bbox']) == 4
        ]
        
        Contextual_bounding_boxes = process_boxes(precise_grounding_positions, image_width, image_height)

        # 构建结果字典
        result = {
            "image_name": "DENTEX_CHALLENGE_2023/xrays/" + image_name,
            "image_width": image_width,
            "image_height": image_height,
            "source": "University of Zurich",
            "modality": "Panoramic X-ray",
            "Dentition Type": "Permanent",
            "Age Classification": "Adult",
            "Question": random.choice(Question_template),
            "Answer": answer_sentence,
            "Precise Grounding Position": precise_grounding_positions,
            "Contextual Grounding Position": Contextual_bounding_boxes,
            "Category": category_list
        }

        # 添加到结果列表
        results.append(result)

    return results


def main():
    # 调用函数处理数据
    input_file = '/home/jinghao/projects/x-ray-VLM/R1/DENTEX_CHALLENGE_2023/train_quadrant_enumeration_disease.json'  # 输入 JSON 文件路径
    results = process_coco_json(input_file)

    # Save the processed data to a JSON file
    output_file = "processed_annotations.json"
    # 将结果写入输出 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()