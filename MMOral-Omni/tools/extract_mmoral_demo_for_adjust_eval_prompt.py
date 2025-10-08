import pandas as pd
import random

# === 参数设置 ===
input_file = "/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid.tsv"     # 输入文件路径
output_file = "/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_demo.tsv"  # 输出文件路径
category_col = "category"    # 分类列名

# === 读取 TSV 数据 ===
df = pd.read_csv(input_file, sep='\t')

# 检查是否存在 category 列
if category_col not in df.columns:
    raise ValueError(f"找不到列 '{category_col}'，请确认文件中有该列。")

# 获取所有类别的集合
categories = df[category_col].unique()
print(f"共有 {len(categories)} 个类别：", categories)

# === 每个类别随机取两个样本 ===
sampled_list = []

for cat in categories:
    cat_df = df[df[category_col] == cat]
    # 小于2个时，全部取出
    n = min(2, len(cat_df))
    sampled = cat_df.sample(n=n, random_state=random.randint(0, 9999))
    sampled_list.append(sampled)

# 合并所有采样结果
sampled_df = pd.concat(sampled_list)

# === 保存新文件 ===
sampled_df.to_csv(output_file, sep='\t', index=False)

print(f"已生成采样文件: {output_file}")
