import json
import os
import random
from PIL import Image
import json
import re

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
    categories = {cat['id']: cat['name'] for cat in data['categories']}

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
        
        # 过滤掉 1.jpg-199.jpg 因为他们的标注包含很大 noise
        if re.fullmatch(r"(?:[1-9]|[1-9]\d|1\d\d)\.jpg", image_name):
            continue

        # 构建 Answer 句子
        answers = []
        first_round_answers = []
        second_round_answers = []
        category = []
        for box in boxes:
            # tooth_id = f"{categories_1.get(box['category_id_1'], '')}{categories_2.get(box['category_id_2'], '')}"
            
            tooth_id = box['tooth_id']
            bbox = box['bbox']
            disease_name = categories.get(box['category_id'], '')
            category.append(disease_name)
            if disease_name == 'Obturation':
                continue
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            # import pdb; pdb.set_trace()
            # answers.append(f"Tooth #{tooth_id} has {disease_name.lower()}.")

            box = f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"
            
            # 构建 answer_sentence 和 precise_grounding_positions
            if disease_name in ['Prosthetic restoration', 'Orthodontic device', 'Surgical device']:
                answers.append(f"The region <box>[{box}]</box> has {disease_name.lower()}.")
                first_round_answers.append(f"The region <box>[{box}]</box> has a {disease_name.lower()}.")
            elif disease_name == 'Impacted tooth':
                answers.append(f"Tooth {tooth_id} is impacted.")
                first_round_answers.append(f"Tooth {tooth_id} is impacted.")
            elif disease_name in ['Implant']:
                answers.append(f"The region <box>[{box}]</box> has a {disease_name.lower()}.")
                first_round_answers.append(f"The region <box>[{box}]</box> has a {disease_name.lower()}.")
            elif disease_name in ['Bone resorbtion']:
                answers.append(f"A {disease_name.lower()} near tooth {tooth_id}.")
                second_round_answers.append(f"A {disease_name.lower()} near tooth {tooth_id}.")
            else:
                if tooth_id != None:
                    answers.append(f"A {disease_name.lower()} at tooth {tooth_id}.")
                    second_round_answers.append(f"A {disease_name.lower()} at tooth {tooth_id}.")
                else:
                    answers.append(f"The region <box>[{box}]</box> has a {disease_name.lower()}.")
                    second_round_answers.append(f"The region <box>[{box}]</box> has a {disease_name.lower()}.")

                # if disease_name =="Root fragment":
                #     print(second_round_answers[-1])

            # if disease_name == "Prosthetic restoration" or disease_name == "Caries":
            #     disease_name = "caries"
            #     answers.append(f"Tooth #{tooth_id} has {disease_name}.")
            # elif disease_name == "Impacted":
            #     disease_name = "impacted"
            #     answers.append(f"Tooth #{tooth_id} is {disease_name}.")
            # elif disease_name == "Periapical Lesion":
            #     disease_name = "periapical lesion"
            #     answers.append(f"Tooth #{tooth_id} has {disease_name}.")
        
        # 将所有 box 的信息拼接成一个句子
        answer_sentence = " ".join(answers)
        first_answer_sentence = " ".join(first_round_answers)
        second_answer_sentence = " ".join(second_round_answers)

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
            "image_name": "Dental_Conditions_Detection_2025_Romania/all/images/" + image_name,
            "image_width": image_width,
            "image_height": image_height,
            "source": "Iuliu Hațieganu University of Medicine and Pharmacy",
            "modality": "Panoramic X-ray",
            "Dentition Type": "Permanent",
            "Age Classification": "Adult",
            "Question": random.choice(Question_template),
            "Full Answer": first_answer_sentence + ' ' + second_answer_sentence if second_answer_sentence else first_answer_sentence,
            "First Round Answer": first_answer_sentence, 
            "Second Round Answer": second_answer_sentence,
            "Precise Grounding Position": precise_grounding_positions,
            "Contextual Grounding Position": Contextual_bounding_boxes,
            "Category": list(category),
        }

        # 添加到结果列表
        results.append(result)

    return results


def main():
    # 调用函数处理数据
    input_file = '/home/jinghao/projects/x-ray-VLM/R1/Dental_Conditions_Detection_2025_Romania_all_with_ToothID.json'  # 输入 JSON 文件路径
    results = process_coco_json(input_file)

    # Save the processed data to a JSON file
    output_file = "processed_annotations.json"
    # 将结果写入输出 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()