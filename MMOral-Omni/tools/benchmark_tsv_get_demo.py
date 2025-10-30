import csv
import sys
csv.field_size_limit(sys.maxsize)

# 输入和输出文件路径

input_file = "/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality.tsv"
output_file = "/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality_480_490.tsv"


# 起始与结束行号（1-based，包含头部后的数据行）
start_line = 480
end_line = 490

with open(input_file, "r", encoding="utf-8", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.reader(infile, delimiter="\t")
    writer = csv.writer(outfile, delimiter="\t")
    
    # 读取表头并写入新文件
    header = next(reader)
    writer.writerow(header)
    
    # 从第 2 行开始计数（因为第 1 行是表头）
    for i, row in enumerate(reader, start=2):
        # 判断是否在指定行区间内
        if start_line <= i <= end_line:
            writer.writerow(row)
        elif i > end_line:
            break

print(f"已将表头与第 {start_line}-{end_line} 行保存到文件: {output_file}")