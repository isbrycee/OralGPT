import pandas as pd

def process_tsv(tsv_file_path, target_category):
    """
    处理 TSV 文件并输出所需信息。
    参数：
        tsv_file_path (str): TSV 文件路径
        target_category (str): 指定的 category 类别名
    """
    # 1. 读取 TSV 文件
    df = pd.read_csv(tsv_file_path, sep='\t')

    # 2. 打印所有表头
    print("🧾 表头（columns）:")
    print(list(df.columns))
    print("-" * 50)

    # 3. 打印 category 列的唯一集合
    if 'category' in df.columns:
        category_set = set(df['category'].dropna().unique())
        print("📂 category 列的唯一值集合：")
        print(category_set)
    else:
        print("❌ 没有找到名为 'category' 的列")
        return
    print("-" * 50)

    # 4. 打印 index 列的数值范围
    if 'index' in df.columns:
        index_min = df['index'].min()
        index_max = df['index'].max()
        print(f"🔢 index 列的范围: [{index_min}, {index_max}]")
    else:
        print("❌ 没有找到名为 'index' 的列")
        return
    print("-" * 50)

    # 5. 打印 TSV 的行数
    print(f"📄 TSV 的总行数: {len(df)}")
    print("-" * 50)

    # 6. 打印指定类别的第一个 case 的所有值（除去 image 列）
    target_rows = df[df['category'] == target_category]

    if target_rows.empty:
        print(f"⚠️ 没有找到 category 为 '{target_category}' 的行。")
        return
    else:
        first_case = target_rows.iloc[0]

        print(f"🎯 category='{target_category}' 的第一个 case 行：")
        for col, val in first_case.items():
            if col != 'image':
                print(f"{col}: {val}")

# 示例调用
process_tsv('/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizeAlphaDent_resizePINormality_filterUnableToAnalysis.tsv', 
            "II_Dx-I,Cancer")
