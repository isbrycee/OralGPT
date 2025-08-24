import os
import json
import cv2
import numpy as np

def visualize_annotations(image_dir, json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_name = data.get("file_name")
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"❌ 找不到图片: {img_path}")
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 无法读取图片: {img_path}")
            continue

        # 遍历 properties
        properties = data.get("properties", {})
        if len(properties['Location']) == 0:
            continue
        for key, value in properties.get("Location", {}).items():
            # 绘制 bbox
            if "bbox" in value:
                for bbox in value["bbox"]:
                    x, y, w, h = bbox
                    x_min, y_min = int(x), int(y)
                    x_max, y_max = int(x + w), int(y + h)
                    # x_min, y_min, x_max, y_max = bbox
                    # x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    cx, cy, w, h = bbox
                    x_min = int(cx - w / 2)
                    y_min = int(cy - h / 2)
                    x_max = int(cx + w / 2)
                    y_max = int(cy + h / 2)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, key, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 绘制 segmentation
            if "segmentation" in value and value["segmentation"]:
                for seg in value["segmentation"]:
                    if not seg:  # 空 list 就跳过
                        continue
                    # seg 是一维 [x1,y1,x2,y2,...]
                    seg_np = np.array(seg, dtype=np.int32).reshape(-1, 2)  # 变成 (N,2)
                    seg_np = seg_np.reshape(-1, 1, 2)  # OpenCV 需要 (N,1,2)
                    cv2.polylines(img, [seg_np], isClosed=True, color=(0, 0, 255), thickness=2)
                    # cv2.fillPoly(img, [seg_np], color=(0, 0, 255))


        # ========= 2. 绘制分类标签 =========
        class_y = 25  # 初始显示位置
        for cls_name, cls_value in properties.get("Classification", {}).items():
            if isinstance(cls_value, dict) and "score" in cls_value:
                score = cls_value["score"]
                text = f"{cls_name}: {score:.2f}"
                cv2.putText(img, text, (20, class_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)
                class_y += 25  # 每行往下移

        # 输出路径
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)
        print(f"✅ 保存可视化结果: {out_path}")


if __name__ == "__main__":
    image_dir = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-images"   # 输入图片文件夹路径
    json_dir = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-jsons-0822"     # 输入 JSON 文件夹路径
    output_dir = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-images-visual"  # 输出结果文件夹路径

    visualize_annotations(image_dir, json_dir, output_dir)
