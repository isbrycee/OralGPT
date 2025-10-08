import json
import random
from collections import defaultdict
from pathlib import Path

def split_data_by_source(input_path, test_size=100):
    # 读取输入JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("输入JSON数据应该是一个列表")
    
    # 统计每个source的数量
    source_counts = defaultdict(int)
    for item in data:
        source = item.get('source')
        if source is None:
            raise ValueError("每个字典必须包含'source'字段")
        source_counts[source] += 1
    
    print("Source统计:")
    for source, count in source_counts.items():
        print(f"{source}: {count}个")
    
    # 计算每个source在test中的分配数量
    total_items = len(data)
    test_distribution = {
        source: round(count / total_items * test_size)
        for source, count in source_counts.items()
    }
    
    # 调整总数可能不等于100的情况
    total_test = sum(test_distribution.values())
    while total_test != test_size:
        diff = test_size - total_test
        # 选择数量最多的source进行调整
        adjust_source = max(test_distribution.items(), key=lambda x: x[1])[0]
        test_distribution[adjust_source] += (1 if diff > 0 else -1)
        total_test = sum(test_distribution.values())
    
    print("\nTest集分配:")
    for source, count in test_distribution.items():
        print(f"{source}: {count}个")
    
    # 按照比例抽取元素
    test_data = []
    train_data = []
    
    # 按source分组
    source_groups = defaultdict(list)
    for item in data:
        source_groups[item['source']].append(item)
    
    # 从每个source中随机抽取指定数量的元素
    for source, count in test_distribution.items():
        group = source_groups[source]
        selected = random.sample(group, count)
        test_data.extend(selected)
        
        # 剩余元素加入train
        for item in group:
            if item not in selected:
                train_data.append(item)
    
    # 打乱顺序
    random.shuffle(test_data)
    random.shuffle(train_data)
    
    # 保存结果
    input_path = Path(input_path)
    test_path = input_path.parent / 'test.json'
    train_path = input_path.parent / 'train.json'
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存:\nTest集: {test_path}\nTrain集: {train_path}")

if __name__ == '__main__':
    import sys
    
    input_json_path = "merged.json"
    split_data_by_source(input_json_path)
