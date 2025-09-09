import json
import time  # 引入 time 模块

# Extract images, tables, and chunk text
# from unstructured.partition.pdf import partition_pdf

# 开始计时
start_time = time.time()

# raw_pdf_elements = partition_pdf(
#     filename="/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_Oral_and_Maxillofacial_Imaging_Diagnostics.pdf",
#     strategy="hi_res",
#     extract_images_in_pdf=True,
#     infer_table_structure=True,
#     chunking_strategy="by_title",
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     image_output_dir_path="/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_figures_Oral_and_Maxillofacial_Imaging_Diagnostics/",
#     languages=["chi_sim"],
# )



# # 转 dict
# elements_as_dict = [el.to_dict() for el in raw_pdf_elements]

# # 保存 json 文件
# with open("textbook_Oral_and_Maxillofacial_Imaging_Diagnostics.json", "w", encoding="utf-8") as f:
#     json.dump(elements_as_dict, f, ensure_ascii=False, indent=2)


import os
import re
import json
import base64
from mineru import MinerU

input_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_pdfs"
output_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_extracted_structured_data"
image_folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_figures_Oral_Mucosal_Diseases"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

miner = MinerU(
    ocr=True,
    language='ch',
    device="gpu"
)

def looks_like_chapter(text, block):
    # 简单判断：短文本 + 含“章” + 字体偏大
    if len(text) <= 15 and "章" in text:
        if block.get("size", 0) > 14:  # 字号条件：>14pt 视为标题
            return True
    return False

def looks_like_section(text, block):
    # 开头是数字，且很短，通常小于 20 字
    if len(text) < 20 and text[0].isdigit() and "." in text[:6]:
        return True
    return False

def looks_like_caption(text):
    # 短文本 + 以 '图' 开头 + 不超过 25 字
    if len(text) <= 25 and text.startswith("图"):
        return True
    return False

def save_image(image_data, save_path):
    """保存 MinerU 的 image block"""
    if isinstance(image_data, str):  # base64
        try:
            img_bytes = base64.b64decode(image_data)
            with open(save_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print("⚠️ 无法保存图片:", e)

def build_structure(doc_struct, book_title="教材"):
    result = {"书名": book_title, "目录": []}
    current_chapter = None
    current_section = None
    img_counter = 0

    for page_idx, page in enumerate(doc_struct.get("pages", []), start=1):
        blocks = page.get("blocks", [])
        for i, block in enumerate(blocks):
            text = block.get("text", "").strip()

            # --- 章节 ---
            if text and looks_like_chapter(text, block):
                current_chapter = {"章节": text, "小节": []}
                result["目录"].append(current_chapter)
                current_section = None
                continue

            # --- 小节 ---
            if text and looks_like_section(text, block):
                current_section = {
                    "标题": text,
                    "段落": [],
                    "表格": [],
                    "图片": []
                }
                if current_chapter:
                    current_chapter["小节"].append(current_section)
                continue

            # --- 表格 ---
            if block.get("type") == "table":
                if current_section:
                    current_section["表格"].append(block)
                continue

            # --- 图片 ---
            if block.get("type") == "image":
                caption = None
                # 在附近找 caption
                for j in range(max(0, i-2), min(len(blocks), i+3)):
                    cand_text = blocks[j].get("text", "").strip()
                    if cand_text and looks_like_caption(cand_text):
                        caption = cand_text
                        break

                img_counter += 1
                img_dir = os.path.join(image_folder, book_title)
                os.makedirs(img_dir, exist_ok=True)
                img_name = f"page{page_idx}_{img_counter}.png"
                img_path = os.path.join(img_dir, img_name)
                save_image(block.get("image"), img_path)

                if current_section:
                    current_section["图片"].append({
                        "path": os.path.relpath(img_path, output_folder),
                        "caption": caption
                    })
                continue

            # --- 段落 ---
            if current_section and text:
                current_section["段落"].append(text)

    return result


for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        print(f"正在处理 {pdf_path}")

        try:
            raw_doc = miner.parse(pdf_path)
            book_name = os.path.splitext(filename)[0]
            structured_doc = build_structure(raw_doc, book_name)

            json_path = os.path.join(output_folder, f"{book_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(structured_doc, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存: {json_path}")

        except Exception as e:
            print(f"❌ 处理 {filename} 出错: {e}")


# 结束计时
end_time = time.time()
print(f"解析 PDF 文件耗时: {end_time - start_time:.2f} 秒")