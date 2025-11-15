import pandas as pd
import re
import os

def has_chinese(text):
    """检测字符串中是否包含中文"""
    if pd.isna(text):
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def clean_tsv(file_path, output_path=None):
    # 读取 TSV 文件
    df = pd.read_csv(file_path, sep='\t')

    print("列名如下：")
    print(df.columns.tolist())

    # 记录要删除的行索引
    drop_indices = []
    for idx, row in df.iterrows():
        for col in df.columns:
            if col == 'image':
                continue
            if has_chinese(row[col]):
                drop_indices.append(idx)
                break

    # 删除包含中文的行
    df_cleaned = df.drop(index=drop_indices)

    # 如果未提供输出路径，则自动生成
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = base + "_cleaned.tsv"

    # 保存为新的 TSV
    # df_cleaned.to_csv(output_path, sep='\t', index=False)

    # 打印统计信息
    print(f"\n已删除 {len(drop_indices)} 行含中文的记录")
    print(f"新文件保存为: {output_path}")
    print(f"新文件总行数: {len(df_cleaned)}")

    # 打印 category 列中的类别数量
    if 'category' in df_cleaned.columns:
        print("\n各类别数量统计：")
        print(df_cleaned['category'].value_counts())
    else:
        print("\n未找到 'category' 列。")

if __name__ == "__main__":
    file_path = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality_filterUnableToAnalysis.tsv'
    clean_tsv(file_path)
