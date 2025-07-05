import json
import os
import random
from PIL import Image

expert_json = "/home/jinghao/projects/x-ray-VLM/R1/Tufts_Dental_Database/Expert/expert.json"
student_json = "/home/jinghao/projects/x-ray-VLM/R1/Tufts_Dental_Database/Student/student.json"

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
    # 计算每个 box 的中心点
    centers = [( (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ) for box in boxes]
    
    # 合并满足条件的 box
    merged_boxes = []
    used = [False] * len(boxes)  # 标记是否已处理过的 box
    
    for i in range(len(boxes)):
        if used[i]:
            continue  # 跳过已经合并的 box
        
        x1, y1, x2, y2 = boxes[i]
        cx1, cy1 = centers[i]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            
            cx2, cy2 = centers[j]
            if abs(cy1 - cy2) < image_height / 5 and abs(cx1 - cx2) < image_width / 8:
                # 合并两个 box
                x1 = min(x1, boxes[j][0])
                y1 = min(y1, boxes[j][1])
                x2 = max(x2, boxes[j][2])
                y2 = max(y2, boxes[j][3])
                used[j] = True
        
        # 添加合并后的 box
        merged_boxes.append([x1, y1, x2, y2])
    
    # 扩大每个 box
    expanded_boxes = []
    for box in merged_boxes:
        x1 = box[0] - 30
        y1 = box[1] - 30
        x2 = box[2] + 30
        y2 = box[3] + 30
        expanded_boxes.append([x1, y1, x2, y2])
    
    return expanded_boxes

def extract_bounding_boxes(polygons):
    """Extract bounding boxes (left, top, right, bottom) from polygons."""
    bounding_boxes = []
    for polygon in polygons:
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        left, top = min(x_coords), min(y_coords)
        right, bottom = max(x_coords), max(y_coords)
        bounding_boxes.append([left, top, right, bottom])
    return bounding_boxes

def process_annotations(expert, student):
    """Process expert and student annotations based on the rules."""
    processed = []
    image_map = {}

    # Combine expert and student annotations into a map
    for annotation in expert:
        image_name = annotation["External ID"]
        if image_name not in image_map:
            image_map[image_name] = {
                "expert": None,
                "student": None
            }
            image_map[image_name]["expert"] = annotation
    for annotation in student:
        image_name = annotation["External ID"]
        image_map[image_name]["student"] = annotation

    # Process each image entry
    for image_name, annotations in image_map.items():
        expert_annotation = annotations["expert"]
        student_annotation = annotations["student"]
        
        # import pdb; pdb.set_trace()
        if expert_annotation["Description"] != "Within normal limits":
            final_annotation = expert_annotation
        elif student_annotation["Description"] != "Within normal limits":
            final_annotation = student_annotation
        else:
            final_annotation = expert_annotation

        # # Use expert annotation if available; otherwise, use student annotation
        # final_annotation = expert_annotation if expert_annotation else student_annotation

        # # Skip images where both annotations are "Within normal limits"
        # if (
        #     expert_annotation and expert_annotation["Description"] == "Within normal limits" and
        #     student_annotation and student_annotation["Description"] == "Within normal limits"
        # ):
        #     continue

        # Extract the required fields

        polygons = final_annotation["Label"]["objects"][0]["polygons"]

        if polygons == 'none':
            bounding_boxes = None
        else:
            bounding_boxes = extract_bounding_boxes(polygons)

        # process bounding_boxes

        # 打开图片
        image_path = "/home/jinghao/projects/x-ray-VLM/R1/Tufts_Dental_Database/Radiographs/" + image_name
        image = Image.open(image_path)
        # 获取宽和高
        width, height = image.size

        if bounding_boxes:
            Contextual_bounding_boxes = process_boxes(bounding_boxes, width, height)
        else:
            Contextual_bounding_boxes = None

        processed_entry = {
            "image_name": "Tufts_Dental_Database/Radiographs/" + image_name,
            "image_width": width, 
            "image_height": height,
            "source": "Tufts University",
            "modality": "Panoramic X-ray",
            "Dentition Type": "Permanent",
            "Age Classification": "Adult",
            "Question": random.choice(Question_template),
            "Answer": final_annotation["Description"],
            "Precise Grounding Position": bounding_boxes,
            "Contextual Grounding Position": Contextual_bounding_boxes
        }
        processed.append(processed_entry)

    return processed

def main():
    # Load the JSON files
    with open(expert_json, "r", encoding="utf-8") as f:
        expert = json.load(f)
    with open(student_json, "r", encoding="utf-8") as f:
        student = json.load(f)

    # Process the annotations
    processed_data = process_annotations(expert, student)

    # Save the processed data to a JSON file
    output_file = "processed_annotations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()