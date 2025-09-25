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

def parse_image_list(s: str):
    """
    将给定的字符串清洗并解析成Python list
    """
    # 去掉换行符
    cleaned = s.replace("\n", "")
    # 把双双引号 "" 变成单个 "
    cleaned = cleaned.replace('""', '"')
    # 再去掉前后多余的空格
    cleaned = cleaned.strip()
    
    # 修复尾随逗号问题：移除数组末尾的逗号
    # 匹配模式：逗号后跟0或多个空格/换行，然后是结束括号
    cleaned = re.sub(r',\s*\]$', ']', cleaned)
    # 用json解析
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析字符串 -> {e}\n清洗后内容: {cleaned}")
    
    return result

def excel_to_tsv(excel_path: str, tsv_path: str):
    """
    将Excel文件转换为TSV文件。
    
    参数:
        excel_path: str - 输入的Excel文件路径
        tsv_path: str - 输出的TSV文件路径
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path, dtype=str)  # 确保每个单元格内容为字符串
    
    # 确保表头一致
    expected_columns = ["index", "image", "question", "answer", "category"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Excel表头应为 {expected_columns}，但实际为 {list(df.columns)}")
    
    # 替换 Excel 内容中的换行符为 \n（字符串），避免保存TSV时换行
    df = df.applymap(lambda x: x.replace("\n", "\\n") if isinstance(x, str) else x)

    # 保存为 tsv，所有字段用引号包裹
    df.to_csv(
        tsv_path, 
        sep="\t", 
        index=False, 
        encoding="utf-8", 
        quoting=csv.QUOTE_ALL
    )

def excel_to_tsv_with_base64(excel_path: str, tsv_path: str):
    """
    将 Excel 转换为 TSV，并将 image 列的路径替换为 base64 list
    """
    df = pd.read_excel(excel_path, dtype=str)

    expected_columns = ["index", "image", "question", "answer", "category"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Excel表头应为 {expected_columns}，但实际为 {list(df.columns)}")

    def process_images(cell_value):
        if pd.isna(cell_value) or not isinstance(cell_value, str):
            return "[]"
        image_paths = parse_image_list(cell_value)
        print(f"Processing {len(image_paths)} images...")
        print(image_paths)
        encoded_list = []
        for img_path in image_paths:
            try:
                img_path = os.path.join(os.path.dirname("/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/hku_cases_textbook_markdown/en_image_pair/"), img_path)
                encoded = encode_image_file_to_base64(img_path)
                encoded_list.append(encoded)
            except Exception as e:
                print(f"⚠️ 处理图像失败: {img_path}, 错误: {e}")
                encoded_list.append("")
        return json.dumps(encoded_list, ensure_ascii=False)

    df["image"] = df["image"].map(process_images)

    # 替换换行符，避免 TSV 出错
    df = df.applymap(lambda x: x.replace("\n", "\\n") if isinstance(x, str) else x)

    df.to_csv(
        tsv_path,
        sep="\t",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL
    )


# 示例调用
if __name__ == "__main__":
    excel_path = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/benchmark_treatment_plan_J.xlsx"   # 输入Excel路径
    tsv_path = "/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_Treatment_Planning.tsv"     # 输出TSV路径
    excel_to_tsv_with_base64(excel_path, tsv_path)
    print(f"保存成功: {tsv_path}")

