import os
import json
from PIL import Image

def process_folders(image_folder, json_folder, output_file):
    results = []
    image_id = 1

    for img_file in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_file)
        
        # 跳过非图像文件
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        name, _ = os.path.splitext(img_file)
        json_file = os.path.join(json_folder, name + ".json")

        if not os.path.exists(json_file):
            print(f"[跳过] 没有找到对应的 JSON: {json_file}")
            continue
        
        # 读取图像尺寸
        with Image.open(img_path) as img:
            width, height = img.size

        # 读取 JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(json_file)

        landmarks = data.get("landmarks", [])

        # 生成 caption
        points = []
        for lm in landmarks:
            title = lm.get("title", "")
            value = lm.get("value", {})
            x, y = value.get("x"), value.get("y")
            points.append(
                f"<points x='{x}' y='{y}' alt='{title}'></points>"
            )

        # 读取 JSON
        with open(json_file.replace("Cephalometric Landmarks/Senior Orthodontists", "CVM Stages"), "r", encoding="utf-8") as f:
            stage_data = json.load(f)

        CVM_stage = stage_data.get("cvm_stage", "Unknown").get("value", "Unknown")

        # 生成 caption
        caption = (
            f"In this lateral cephalometric radiograph, {len(points)} landmarks were detected: \n"
            + ",\n".join(points)
            + ".\n"
            + "Based on the morphological characteristics of the cervical vertebrae (C2–C4) derived from these landmarks, the cervical vertebral maturation (CVM) stage was estimated. The analysis considered vertebral body shape, height-to-width ratios, and the degree of concavity at the lower borders.\n"  
            + f"The prediction indicates that the patient is at **CVM Stage S{CVM_stage}**."
        )

        results.append({
            "image_id": str(image_id),
            "file_name": "Aariz/test/Cephalograms/" + img_file,
            "width": width,
            "height": height,
            'split': 'test',
            "modality": "Cephalometric radiograph",
            "caption": caption
        })

        image_id += 1

    # 保存统一的 json 文件
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"[完成] 已保存到 {output_file}")


if __name__ == "__main__":
    image_folder = "/home/jinghao/projects/x-ray-VLM/RGB/cephalometric_radiographs/Aariz/test/Cephalograms"   # 替换
    json_folder = "/home/jinghao/projects/x-ray-VLM/RGB/cephalometric_radiographs/Aariz/test/Annotations/Cephalometric Landmarks/Senior Orthodontists"     # 替换
    output_file = "merged_annotations_test.json"

    process_folders(image_folder, json_folder, output_file)
