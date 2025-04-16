import json
import os
import base64
import io
from PIL import Image
import csv
from tqdm import tqdm
import random

def filter_ocr_items(data_list):
    return [item for item in data_list if item.get("Category") != "OCR"]

def encode_image_file_to_base64(image_path, target_size=-1):
    image = Image.open(image_path)
    return encode_image_to_base64(image, target_size=target_size)

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
    return ret

def process_json_folder(image_folder_path, json_folder_path, output_tsv_path, target_size=-1):
    """
    Processes JSON files from a folder and generates a .tsv file.
    :param json_folder_path: Path to the folder containing JSON files.
    :param output_tsv_path: Path to output the .tsv file.
    :param target_size: Maximum size for resizing images before encoding.
    """
    index = 1
    rows = []
    cate_list = ['teeth', 'patho', 'his', 'jaw', 'summ']
    img_set = set()
    # Iterate over all JSON files in the folder
    num = 0
    for filename in tqdm(os.listdir(json_folder_path)):
        # if index > 50:
        #     break
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder_path, filename)
            try:
                # Load JSON file
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_name = data.get("file_name", "").strip()  # Assuming the JSON includes an 'image' field
                image_path = os.path.join(image_folder_path, image_name)
                # Encode image to Base64
                if image_path and os.path.isfile(image_path):
                    base64_image = encode_image_file_to_base64(image_path, target_size)
                else:
                    base64_image = ""

                # Extract "Open-End Questions"
                # select qa pairs
                loc_open_end_questions = filter_ocr_items(data['vqa_data']["loc_open_ended"])
                med_open_end_questions = filter_ocr_items(data['vqa_data']["med_open_ended"])
                report_gene_questions = data['vqa_data']["report_generation"]

                if len(med_open_end_questions) < 5:
                    open_end_questions = med_open_end_questions  # 返回整个序列
                    open_end_questions += random.sample(loc_open_end_questions, 5-len(med_open_end_questions))
                else:
                    open_end_questions = random.sample(med_open_end_questions, 4)
                    open_end_questions += random.sample(loc_open_end_questions, 1)

                open_end_questions += random.sample(report_gene_questions, 1)

                num += len(open_end_questions)

                for entry in open_end_questions:
                    question = entry.get("Question", "").strip()
                    answer = entry.get("Answer", "").strip()
                    category = entry.get('Category', 'unknown')
                    # category = random.choice(cate_list)
                    
                    # Append row data
                    if image_name not in img_set:
                        rows.append([index, image_name, base64_image, question, answer, category])
                        img_set.add(image_name)
                    else:
                        rows.append([index, image_name, '', question, answer, category])
                    index += 1

            except Exception as e:
                print(f"Error processing file {json_path}: {e}")
    print(num)
    # Write rows to a .tsv file
    with open(output_tsv_path, 'w', encoding='utf-8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        # Write header
        tsv_writer.writerow(["index", "image_name", "image", "question", "answer", "category"])
        # Write rows
        tsv_writer.writerows(rows)

    print(f"TSV file created at {output_tsv_path}")

# Example usage:
process_json_folder("/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test/MM-Oral-OPG-images",
                    '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test/MM-Oral-OPG-vqa-loc-med', 
                    '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test/MM-Oral-VQA-Open-Ended.tsv')