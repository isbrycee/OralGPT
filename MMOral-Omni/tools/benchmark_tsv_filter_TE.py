import pandas as pd

def clean_tsv(input_file, output_file):
    # 读取 TSV 文件
    df = pd.read_csv(input_file, sep='\t')
    
    # 删除 image 列为空或者空字符串的行
    df = df[df['image'].notna() & (df['image'] != '')]
    
    # 重新排列 index 列，从 1 开始
    df['index'] = range(1, len(df) + 1)
    
    # 保存为新的 TSV 文件
    df.to_csv(output_file, sep='\t', index=False)
    
    # 输出过滤后的行数
    print(f'过滤后 TSV 文件的行数：{len(df)}')

if __name__ == "__main__":
    input_path = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned.tsv'  # 输入文件路径，替换为您的文件名
    output_path = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE.tsv'  # 输出文件路径，替换为您想保存的文件名
    clean_tsv(input_path, output_path)

