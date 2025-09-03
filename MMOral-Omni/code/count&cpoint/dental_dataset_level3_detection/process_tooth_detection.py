import json
import os
from collections import Counter, defaultdict

##把牙齿检测的 COCO 标注文件解析并转换为带统计信息和自动生成文字描述的 JSON 数据集

def process_coco_file(split_name, input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    results = []

    for image in images:
        image_id = f"{split_name}_{image['id']}"
        file_name = image['file_name']
        height = image['height']
        width = image['width']

        # 获取该图像对应的标注
        image_annotations = [ann for ann in annotations if ann['image_id'] == image['id']]

        annotation_details = []
        category_counter = Counter()
        category_summary = defaultdict(list)

        for ann in image_annotations:
            category_name = categories.get(ann['category_id'], 'unknown')
            bbox = ann['bbox']
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            cpoint = [round(cx, 2), round(cy, 2)]

            annotation_details.append({
                'category': category_name,
                'bbox': bbox
            })

            category_counter[category_name] += 1
            category_summary[category_name].append(cpoint)

        # 构造 category_summary 列表
        category_summary_list = [
            {
                'category': cat,
                'count': len(locs),
                'locations': locs
            }
            for cat, locs in category_summary.items()
        ]

        caption = generate_caption(len(annotation_details), category_summary_list)

        results.append({
            'image_id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height,
            'modality': 'Intraoral photography',  # ✅ 新增字段
            'tooth_count': len(annotation_details),
            'annotations': annotation_details,
            'category_count': dict(category_counter),
            'category_summary': category_summary_list,
            'caption': caption
        })

    return results

# ✅ 英文 Caption 生成函数
def generate_caption(tooth_count, category_summary):
    caption = f"This is an intraoral photography image containing {tooth_count} teeth. "
    caption += f"There are {len(category_summary)} tooth types identified, including "

    parts = []
    for item in category_summary:
        cat = item['category']
        count = item['count']
        locs = item['locations']
        # 坐标四舍五入取整数，并加上 <points> 标签
        loc_str = ", ".join([f"<points>({round(x)}, {round(y)})</points>" for x, y in locs])
        plural_s = 's' if count > 1 else ''
        parts.append(f"{count} {cat}{plural_s} located at [{loc_str}]")

    caption += "; ".join(parts) + "."
    return caption

# === 主逻辑 ===
if __name__ == "__main__":
    tasks = [
        {
            'split': 'train',
            'input': './image/count&cpoint/train/_annotations.coco.json',
            'output': './dental_dataset/count&cpoint/train_parsed_annotations.json'
        },
        {
            'split': 'valid',
            'input': './image/count&cpoint/valid/_annotations.coco.json',
            'output': './dental_dataset/count&cpoint/valid_parsed_annotations.json'
        },
        {
            'split': 'test',
            'input': './image/count&cpoint/test/_annotations.coco.json',
            'output': './dental_dataset/count&cpoint/test_parsed_annotations.json'
        },
    ]
    merged_output_path = './dental_dataset/count&cpoint/all_parsed_annotations.json'
    all_results = []

    for task in tasks:
        split = task['split']
        input_path = task['input']
        output_path = task['output']

        split_results = process_coco_file(split, input_path)

        with open(output_path, 'w') as f:
            json.dump(split_results, f, indent=4, ensure_ascii=False)
        print(f"✅ {split.upper()} saved to {output_path}")

        all_results.extend(split_results)

    # 合并所有 split 的输出
    with open(merged_output_path, 'w') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"✅ ALL splits merged and saved to {merged_output_path}")
