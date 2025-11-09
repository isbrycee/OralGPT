import pandas as pd

# 输入文件路径和输出文件路径
input_file = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality_filterUnableToAnalysis.tsv'
output_file = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality_filterUnableToAnalysis_woimg.tsv'

# 读取 TSV 文件
df = pd.read_csv(input_file, sep='\t')

# 移除 'images' 列
df_filtered = df.drop(columns=['image'])

# 保存新的 TSV 文件
df_filtered.to_csv(output_file, sep='\t', index=False)
