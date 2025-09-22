import os
import json
from PIL import Image

# 定义 landmark 对照表
LANDMARK_MAP = {
    1: "Sella",
    2: "Nasion",
    3: "Orbitale",
    4: "Porion",
    5: "Subspinale",
    6: "Supramentale",
    7: "Pogonion",
    8: "Menton",
    9: "Gnathion",
    10: "Gonion",
    11: "Incision inferius",
    12: "Incision superius",
    13: "Upper lip",
    14: "Lower lip",
    15: "Subnasale",
    16: "Soft tissue pogonion",
    17: "Posterior nasal spine",
    18: "Anterior nasal spine",
    19: "Articulare",
}

def process_folders(image_folder, txt_folder, output_file):
    results = {"annotations": []}
    image_id = 1

    for img_file in sorted(os.listdir(image_folder)):
        if not img_file.lower().endswith(('.jpg', '.bmp', '.png')):
            continue

        name, _ = os.path.splitext(img_file)
        txt_file = os.path.join(txt_folder, name + ".txt")

        if not os.path.exists(txt_file):
            print(f"[跳过] 对应的 TXT 没找到: {txt_file}")
            continue

        # 读取图像尺寸
        img_path = os.path.join(image_folder, img_file)
        with Image.open(img_path) as img:
            width, height = img.size

        # 读取 txt 文件
        points = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # 空行跳过
                    continue
                try:
                    x, y = line.split(",")
                    x, y = int(x), int(y)
                except ValueError:
                    print(f"[警告] {txt_file} 第 {idx} 行格式错误: {line}")
                    continue

                # 匹配 landmark 名称
                title = LANDMARK_MAP.get(idx, f"L{idx}").lower()
                points.append(f"<points>(x='{x}', y='{y}', alt='{title}')</points>")


        # 生成 caption
        caption = (
            f"In this lateral cephalometric radiograph, {len(points)} landmarks were detected: \n"
            + ",\n".join(points)
        )

        results["annotations"].append({
            "image_id": str(image_id),
            "file_name": "dental-cepha-dataset-pku/" + img_file,
            "width": width,
            "height": height,
            "modality": "Cephalometric radiograph",
            "caption": caption
        })

        image_id += 1

    # 保存统一 json
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"[完成] 结果已保存到 {output_file}")


if __name__ == "__main__":
    image_folder = "/home/jinghao/projects/x-ray-VLM/RGB/cephalometric_radiographs/dental-cepha-dataset-pku/image"   # TODO: 修改为你的图像路径
    txt_folder = "/home/jinghao/projects/x-ray-VLM/RGB/cephalometric_radiographs/dental-cepha-dataset-pku/doctor1"       # TODO: 修改为你的 TXT 路径
    output_file = "merged_annotations.json"

    process_folders(image_folder, txt_folder, output_file)
