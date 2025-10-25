import json
import re
from PIL import Image
import os
import random

replacement_questions_gingivitis = [
    "This is an intraoral photograph. Can you identify the regions where gingivitis is observed? Please provide the regions as bounding boxes in <box></box> format.",
    "This is an intraoral photograph. Which regions show signs of gingivitis? Please represent the regions as bounding boxes using the <box></box> tag.",
    "Based on this intraoral photograph, which teeth are associated with gingivitis? Please represent the regions as bounding boxes using the <box></box> tag."
    "Locate the areas of gingival inflammation in this intraoral photograph. Please provide the coordinates as bounding boxes in <box></box> format.",
    "Identify the regions affected by gingivitis in this intraoral photograph. Please provide the bounding box coordinates in <box></box> format."
]

replacement_questions_AlphaDent = [
    "This is an intraoral photograph. Can you identify the regions where abrasion, filling, crown, or caries is observed? Please provide the regions as bounding boxes in <box></box> format.",
    "This is an intraoral photograph. Which regions show signs of the abrasion, filling, crown, or caries? Please represent the regions as bounding boxes using the <box></box> tag.",
    "Based on this intraoral photograph, which teeth are associated with abrasion, filling, crown, or caries? Please represent the regions as bounding boxes using the <box></box> tag."
    "Locate the areas of abrasion, filling, crown, or caries in this intraoral photograph. Please provide the coordinates as bounding boxes in <box></box> format.",
    "Identify the regions affected by abrasion, filling, crown, or caries in this intraoral photograph. Please provide the bounding box coordinates in <box></box> format."
]

sentences_for_PI = [
    "Please provide the cell nuclei as bounding boxes in <box></box> format.",
    "Please represent the cell nuclei as bounding boxes using the <box></box> tag.",
    "Please provide the coordinates as bounding boxes in <box></box> format.",
    "Please represent the cell nuclei using bounding boxes in the <box></box> format.",
    "Please provide the bounding box coordinates in <box></box> format."
]

def resize_image(image_path):
    img = Image.open(image_path)
    image_path_name = image_path.split('/home/jinghao/projects/x-ray-VLM/RGB/')[-1]
    w, h = img.size
    max_edge = max(w, h)
    if max_edge > 768:
        if w > h:
            new_w = 768
            new_h = int(h * 768 / w)
        else:
            new_h = 768
            new_w = int(w * 768 / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # dirname, filename = os.path.split(image_path)
        # new_filename = f"resized_{filename}"
        # new_path = os.path.join(dirname, new_filename)
        img.save(image_path)
        scale_w = new_w / w
        scale_h = new_h / h
        return image_path_name, scale_w, scale_h
    else:
        return image_path_name, 1.0, 1.0

def scale_boxes(gpt_value, scale_w, scale_h):
    box_pattern = re.compile(r'<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>')
    def replace_box(match):
        x1, y1, x2, y2 = map(int, match.groups())
        x1 = int(x1 * scale_w)
        y1 = int(y1 * scale_h)
        x2 = int(x2 * scale_w)
        y2 = int(y2 * scale_h)
        return f'<box>[{x1},{y1},{x2},{y2}]</box>'
    new_value = box_pattern.sub(replace_box, gpt_value)
    return new_value

def process_json_list(json_list):
    modified_count = 0
    for item in json_list:
        if 'images' in item and item['images']:
            orig_path = os.path.join("/home/jinghao/projects/x-ray-VLM/RGB/", item['images'][0])
            new_path, scale_w, scale_h = resize_image(orig_path)

            if new_path != orig_path:
                item['images'][0] = new_path
                if '<box>' not in item['conversations'][1]['value']:
                    continue
                for conv in item['conversations']:
                    if conv['from'] == 'human':
                        old_question = conv['value']
                        new_sentence = random.choice(sentences_for_PI)
                        conv['value'] = old_question + ' ' + new_sentence
                    if conv['from'] == 'gpt':
                        old_len = len(conv['value'])
                        conv['value'] = scale_boxes(conv['value'], scale_w, scale_h)
                        new_len = len(conv['value'])
                        modified_count += 1
                        print(f"Modified gpt conversation: original length {old_len}, new length {new_len}")

    print(f"Total modified conversations: {modified_count}")
    return json_list

if __name__ == '__main__':
    input_file = '/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/5.1_sft_histopathologicalImage_Diagnosis_4datasets_shareGPT.json'  # 需替换为你的输入 json 文件名
    output_file = '/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/5.1_sft_histopathologicalImage_Diagnosis_4datasets_shareGPT_resize.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified_data = process_json_list(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)

