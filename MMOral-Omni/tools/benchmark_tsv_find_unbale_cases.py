import pandas as pd



# ======== 修改这里，替换为你的输入文件路径 ========
input_file = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality.tsv'
output_file = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality_filterUnableToAnalysis.tsv'
# ===================================================

# 读取 TSV 文件
df = pd.read_csv(input_file, sep='\t', dtype=str)

# 确保存在 'answer' 列
if 'answer' not in df.columns:
    raise ValueError("文件中未找到 'answer' 列")

# 找出包含 'unbale' 的行
mask = df['answer'].str.contains('Unable to analyze', case=False, na=False)

# 删除这些行
filtered_df = df[~mask]

# 保存为新的 TSV 文件
filtered_df.to_csv(output_file, sep='\t', index=False)

# 打印新文件行数
print(f"新文件 '{output_file}' 保存成功！")
print(f"新 TSV 文件的行数为：{len(filtered_df)}")