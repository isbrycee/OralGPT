import json

import json
import os

def count_bbox_values(bbox_data):
    """统计单个bbox字段中的bbox数量"""
    if isinstance(bbox_data, list):
        if not bbox_data:
            return 0
        # 检查是否为嵌套列表（二维）
        if isinstance(bbox_data[0], list):
            return len(bbox_data)
        else:
            # 单个四元组列表
            return 1
    return 0

def count_bbox_in_obj(obj):
    """递归遍历对象，统计所有bbox字段的数量"""
    count = 0
    if isinstance(obj, dict):
        # 检查当前字典是否有bbox字段
        if 'bbox' in obj:
            count += count_bbox_values(obj['bbox'])
        # 递归处理所有值
        for value in obj.values():
            count += count_bbox_in_obj(value)
    elif isinstance(obj, list):
        for item in obj:
            count += count_bbox_in_obj(item)
    return count

def process_folder(folder_path):
    """处理文件夹中的所有JSON文件，返回总bbox数量"""
    total = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    properties = data.get('properties', {})
                    # 遍历properties中的每个键值对
                    for prop_key, prop_value in properties.items():
                        if isinstance(prop_value, list):
                            # 遍历列表中的每个元素并递归统计
                            for item in prop_value:
                                total += count_bbox_in_obj(item)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    return total


# ################## for computing number of visual attributes  ##################
# 使用示例
folder_path = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/train/MM-Oral-OPG-visual-attributes'
total_bbox = process_folder(folder_path)
print(f"总 Attributes 数量: {total_bbox}")

# ################## for computing number of vqa  ##################

def count_vqa_elements(json_folder):
    total = 0
    target_keys = ['loc_closed_ended', 'loc_open_ended', 'med_closed_ended', 'med_open_ended']
    # target_keys = ['loc_open_ended',  'med_open_ended']
    
    for filename in os.listdir(json_folder):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(json_folder, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            vqa_data = data.get('vqa_data', {})
            current_count = 0
            for key in target_keys:
                items = vqa_data.get(key, [])
                if isinstance(items, list):  # 确保值为列表类型
                    current_count += len(items)
            
            total += current_count
        
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    return total

# 使用示例
json_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/test/MM-Oral-OPG-vqa-loc-med"
result = count_vqa_elements(json_folder)
print(f"总 single vqa 数量: {result}")


# ################## for computing number of conversation ##################

def sum_conversations_length(folder_path):
    total = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        conversations = data.get('conversations', [])
                        if isinstance(conversations, list):
                            total += len(conversations)
                        else:
                            print(f"⚠️ {file_path}: conversations字段类型不是列表")
                except json.JSONDecodeError:
                    print(f"❌ {file_path}: 文件格式错误，无法解析")
                except Exception as e:
                    print(f"⚠️ {file_path}: 发生错误 - {str(e)}")
    return total


# folder_path = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/train/MM-Oral-OPG-multi-turn-conv"
# result = sum_conversations_length(folder_path)
# print(f"📊 conversations总长度: {result}")