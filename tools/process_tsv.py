import pandas as pd

# 原始 TSV 文件路径
original_file = "/home/jinghao/projects/x-ray-VLM/dataset/MMOral-Bench/MM-Oral-VQA-Closed-Ended.tsv"

# 新的 TSV 文件路径
new_file = "/home/jinghao/projects/x-ray-VLM/dataset/MMOral-Bench/MM-Oral-VQA-Open-Ended.tsv"

# 输出合并后的 TSV 文件路径
output_file = "updated_file.tsv"

# 读取原始 TSV 文件和新的 TSV 文件
original_df = pd.read_csv(original_file, sep="\t")
original_df = pd.read_csv(new_file, sep="\t")

# 删除 'image' 列
if 'image' in original_df.columns:
    original_df = original_df.drop(columns=['image'])

# 保存结果为新的 TSV 文件
original_df.to_csv(output_file, sep="\t", index=False)

print(f"合并后的文件已保存到: {output_file}")
