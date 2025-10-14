import csv
import sys
from collections import Counter

def count_categories_from_tsv(file_path):
    # 提高 CSV 模块的最大字段长度限制
    csv.field_size_limit(sys.maxsize)

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        categories = [row['category'] for row in reader if 'category' in row and row['category']]

    category_counts = dict(Counter(categories))
    return category_counts

if __name__ == "__main__":
    file_path = '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral-Omni-Bench.tsv'  # 替换为你的 TSV 文件路径
    result = count_categories_from_tsv(file_path)
    print(result)
