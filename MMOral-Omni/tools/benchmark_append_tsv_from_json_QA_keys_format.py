import pandas as pd
import csv
import json
import os
import io
import numpy as np
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
import re
import sys

csv.field_size_limit(sys.maxsize)

def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img

def encode_image_file_to_base64(image_path, target_size=-1, fmt='JPEG'):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size, fmt=fmt)

def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret

def json_to_tsv(json_path, tsv_path, output_path=None):
    # 如果没有指定输出，覆盖原 tsv 文件
    assert output_path is not None, "请指定 output_path，避免覆盖原文件"

    # 读取已有 tsv，找出最后一个 index
    last_index = 0
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            last_index = max(last_index, int(row["index"]))

    # 读取 json 文件
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 追加新数据
    new_rows = []
    
    for item in json_data:
        split = item.get("split", "train")
        if split == "train":
            continue
        question = item.get("question", "").strip()
        question = question.replace("\n", "\\n").strip()

        answer = item.get("caption", "").strip()
        answer = answer.replace("\n", "\\n").strip()

        category = item.get("category", "cepha")
        img_path = item.get("file_name", "")
        img_path = os.path.join("/home/jinghao/projects/x-ray-VLM/RGB/cephalometric_radiographs", img_path)
        last_index += 1
        new_rows.append({
            "index": str(last_index),
            "image": encode_image_file_to_base64(img_path),
            "question": question,
            "answer": answer,
            "category": category
        })

    # 合并原数据和新增数据，写回 TSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["index", "image", "question", "answer", "category"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_ALL )  # 关键设置：强制所有字段加双引号
        
        writer.writeheader()
        writer.writerows(rows + new_rows)

    print(f"已生成输出 TSV 文件: {output_path}")

# 示例调用
json_to_tsv(json_path="/home/jinghao/projects/x-ray-VLM/RGB/cephalometric_radiographs/cephalometric_radiographs_Aariz_caption_data_question.json", 
            tsv_path="/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc.tsv",
            output_path="/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha.tsv")

