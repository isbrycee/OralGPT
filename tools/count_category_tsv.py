import csv
from collections import Counter
import sys

def count_categories(file_path):
    # 增大字段大小限制
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 2

    # 用于存储类别计数
    category_counts = Counter()
    
    # 打开并读取 TSV 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        # 遍历每一行，统计 `category` 列
        for row in reader:
            # 获取 `category` 列的值，并按照逗号分隔
            categories = row['category'].split(',')
            for category in categories:
                category = category.strip()  # 去除空格
                if category:  # 确保类别非空
                    category_counts[category] += 1
    
    # 打印所有类别及其数量
    for category, count in category_counts.items():
        print(f"{category}: {count}")


# 替换为你的 TSV 文件路径
file_path = '/home/jinghao/LMUData/MM-Oral-VQA-Open-Ended.tsv'
count_categories(file_path)
