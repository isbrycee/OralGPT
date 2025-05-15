import csv

def count_category(file_path):
    category_counts = {}
    
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        
        for row in reader:
            category = row['category']
            category_counts[category] = category_counts.get(category, 0) + 1
    
    return category_counts

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python script.py <输入文件路径>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = count_category(input_file)
    
    for category, count in result.items():
        print(f"{category}\t{count}")
