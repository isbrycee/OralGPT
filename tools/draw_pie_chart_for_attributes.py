import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
import plotly.express as px
import plotly.offline as offline
import plotly.io as pio
import colorsys
import plotly.graph_objects as go
import pandas as pd

def count_bboxes_in_json(json_data):
    category_counts = defaultdict(int)
    
    def traverse(data, parent_key_path=None):
        parent_key_path = parent_key_path or []
        
        if isinstance(data, dict):
            if 'bbox' in data:
                bbox_value = data['bbox']
                category = determine_category(parent_key_path, data)
                # print(category)
                if category == 'Periapical lesions':
                    sub_cate = data.get('type', None)
                    if sub_cate:
                        if isinstance(sub_cate, list):
                            sub_cate = sub_cate[0]
                        category = f'Periapical lesions {sub_cate}'
                    else:
                        category = 'Periapical lesion'
                if category == 'Periapical lesion':
                    sub_cate = data.get('type', None)
                    if sub_cate:
                        if '[' in sub_cate:
                            sub_cate = sub_cate[2:-2]
                        category = f'Periapical lesion {sub_cate}'
                    else:
                        category = 'Periapical lesion'
                process_bbox(bbox_value, category)
            
            for key, value in data.items():
                new_path = parent_key_path + [key]
                traverse(value, new_path)
                
        elif isinstance(data, list):
            for item in data:
                traverse(item, parent_key_path)
    
    def determine_category(path, current_dict):
        if 'tooth_id' in current_dict:
            return f'Tooth_{current_dict['tooth_id']}'
        if 'quadrant' in current_dict:
            return current_dict['quadrant']
        if 'JawBones' in path:
            for i in reversed(range(len(path))):
                if path[i] == 'conditions' and i+1 < len(path):
                    return path[i+1]
            return 'JawBones'
        if 'conditions' in path:
            for i in reversed(range(len(path))):
                if path[i] == 'conditions' and i+1 < len(path):
                    return path[i+1]
        return path[-1] if path else 'Unknown'
    
    def process_bbox(bbox_data, category):
        nonlocal category_counts  # 添加nonlocal声明
        if isinstance(bbox_data, list):
            if all(isinstance(item, list) for item in bbox_data):
                for sub_bbox in bbox_data:
                    if len(sub_bbox) >= 4:
                        category_counts[category] += 1
            elif len(bbox_data) >= 4:
                category_counts[category] += 1
    
    traverse(json_data.get('properties', {}))
    return dict(category_counts)

def process_json_folder(folder_path):
    total_counts = defaultdict(int)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    counts = count_bboxes_in_json(data)
                    for k, v in counts.items():
                        total_counts[k] += v
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return dict(total_counts)

def categorize_data(original_dict, cate_divide_map):
    # 创建反向映射字典，子类到父类的映射
    reverse_map = {}
    for parent, children in cate_divide_map.items():
        for child in children:
            reverse_map[child] = parent
    
    # 初始化结果字典
    result = {parent: {} for parent in cate_divide_map}
    
    # 遍历原始字典中的每个条目，分配到对应的父类下
    for key, value in original_dict.items():
        parent = reverse_map.get(key)
        if parent:
            result[parent][key] = value
    
    return result

# 使用示例
if __name__ == "__main__":
    json_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/train/MM-Oral-OPG-visual-attributes'
    result_train = process_json_folder(json_folder)
    json_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/test/MM-Oral-OPG-visual-attributes'
    result_test = process_json_folder(json_folder)
    print("Bounding Box Category Counts:")
    num = 0
    all_dict = {}
    for category, count in sorted(result_train.items()):
        all_dict[category] = count + result_test[category]
        print(f"{category}: {count}")
        num += count

    all_dict['Periapical lesions Granuloma'] += all_dict['Periapical lesion']
    # all_dict.pop('Retained root')
    all_dict.pop('Periapical lesion')

    # print(all_dict)

    cate_divide_map = {
        'Jaw': ['Bone loss', 'Mandibular canal', 'Maxillary sinus'],
        'Tooth':["Tooth_11", "Tooth_12", "Tooth_13", "Tooth_14", "Tooth_15", "Tooth_16", "Tooth_17", "Tooth_18",
                 "Tooth_21", "Tooth_22", "Tooth_23", "Tooth_24", "Tooth_25", "Tooth_26", "Tooth_27", "Tooth_28",
                 "Tooth_31", "Tooth_32", "Tooth_33", "Tooth_34", "Tooth_35", "Tooth_36", "Tooth_37", "Tooth_38",
                 "Tooth_41", "Tooth_42", "Tooth_43", "Tooth_44", "Tooth_45", "Tooth_46", "Tooth_47", "Tooth_48"],
        'Quadrant': ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4',],
        'Historical treatments': ['Crown', 'Implant', 'Filling', 'Root canal treatment'],
        'Pathological Findings': ['Caries', 'Deep caries', 'Periapical lesion', 'Periapical lesions Abscess', 
                                  'Periapical lesions Cyst', 'Periapical lesions Granuloma', 'Impacted tooth', 'Missing teeth',
                                  'Retained root'],
    }

    result = categorize_data(all_dict, cate_divide_map)
    print(result)

    data = {
        'Jaw': {'Bone loss': 5657, 'Mandibular canal': 32841, 'Maxillary sinus': 24530},
        'Tooth': {'Tooth #11': 19028, 'Tooth #12': 19037, 'Tooth #13': 19278, 'Tooth #14': 17954, 'Tooth #15': 17010, 'Tooth #16': 17279, 'Tooth #17': 18553, 'Tooth #18': 11842, 'Tooth #21': 16880, 'Tooth #22': 17348, 'Tooth #23': 16295, 'Tooth #24': 16901, 'Tooth #25': 15130, 'Tooth #26': 15986, 'Tooth #27': 16472, 'Tooth #28': 14125, 'Tooth #31': 17991, 'Tooth #32': 18144, 'Tooth #33': 19095, 'Tooth #34': 17643, 'Tooth #35': 19078, 'Tooth #36': 17018, 'Tooth #37': 17108, 'Tooth #38': 14141, 'Tooth #41': 18960, 'Tooth #42': 19627, 'Tooth #43': 19326, 'Tooth #44': 19346, 'Tooth #45': 18745, 'Tooth #46': 17295, 'Tooth #47': 17869, 'Tooth #48': 13778},
        'Quadrant': {'Quadrant #1': 20385, 'Quadrant #2': 20450, 'Quadrant #3': 20375, 'Quadrant #4': 20412},
        'Pathological Findings': {
            'Impacted tooth': 27505,  # 手动调整顺序
            'Caries': 21855,
            'Missing teeth': 14342,
            'Periapical lesions': 10375,
            'Deep caries': 7338,
            'Retained root': 5887
            # 'Periapical lesions Granuloma': 7242,
            # 'Periapical lesions Abscess': 2324,
            # 'Periapical lesions Cyst': 809,
            
        },
        'Historical Treatments': {'Filling': 73237, 'Crown': 27259, 'Root canal treatment': 20043, 'Implant': 3553, },
        
    }
    # Prepare data for the sunburst plot
    labels = ['MMOral-Attribute'] + list(data.keys()) + [item for sublist in data.values() for item in sublist.keys()]
    parents = [''] + ['MMOral-Attribute'] * len(data) + [parent for parent, sublist in data.items() for _ in sublist]
    values = [sum(sum(sublist.values()) for sublist in data.values())] + [sum(sublist.values()) for sublist in data.values()] + [value for sublist in data.values() for value in sublist.values()]
    
    print(labels)
    print(parents)
    print(values)

    # Create the sunburst plot
    fig = px.sunburst(
        names=labels,
        parents=parents,
        values=values,
        # title="Attribute Distribution",
        branchvalues='total',
        width=1000,
        height=1000,
    )

    # Update layout for better readability
    fig.update_layout(
        sunburstcolorway=[ "#FFA15A", "#636EFA", "#AB63FA", "#EF553B", "#00CC96",],
        margin=dict(t=0, l=0, r=0, b=0),
        # uniformtext=dict(minsize=12, mode='hide'),
        font_family="Times New Roman",  # 全局字体
    )

    text_templates = [
        '%{label}' if label == "MMOral-Attribute" else 
        ('%{label} %{percentParent:.1%}' if (value/sum(values) >= 0.001) else '')
        for label, value in zip(labels, values)
    ]
    special_font_sizes = {
        "MMOral-Attribute": 30,
        "Jaw": 20,
        "Historical Treatments": 15,
        "Pathological Findings": 15,
        "Tooth": 20,
        "Quadrant": 20
    }
    # Update traces for better text readability
    fig.update_traces(
        texttemplate=text_templates,
        # textinfo="label+percent entry",
        insidetextorientation='radial',
        sort=False, selector=dict(type='sunburst'),
        textfont=dict(
            family="Times New Roman",
            size=[special_font_sizes.get(label, 12) for label in labels],
            color="black"
            ),
        # rotation=135  # Rotate the chart by 90 degrees
        )


    # Save the plot as a PNG file
    pio.write_image(fig, 'MMOral-Attribute_dist.png', format='png', scale=1)
