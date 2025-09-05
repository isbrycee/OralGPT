import json
import time  # 引入 time 模块

# Extract images, tables, and chunk text
from unstructured.partition.pdf import partition_pdf

# 开始计时
start_time = time.time()

raw_pdf_elements = partition_pdf(
    filename="/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_Oral_and_Maxillofacial_Imaging_Diagnostics.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path="/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/textbook_figures_Oral_and_Maxillofacial_Imaging_Diagnostics/",
    languages=["chi_sim"],
)

# 结束计时
end_time = time.time()

# 转 dict
elements_as_dict = [el.to_dict() for el in raw_pdf_elements]

# 保存 json 文件
with open("textbook_Oral_and_Maxillofacial_Imaging_Diagnostics.json", "w", encoding="utf-8") as f:
    json.dump(elements_as_dict, f, ensure_ascii=False, indent=2)


# 打印解析时间
print(f"解析 PDF 文件耗时: {end_time - start_time:.2f} 秒")