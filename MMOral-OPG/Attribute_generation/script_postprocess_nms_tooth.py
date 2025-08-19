import os
import json
import glob
import argparse
from typing import Dict, List, Any

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个边界框的IoU值"""
    # 获取两个框的坐标 (已经是xyxy格式)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集区域的坐标
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # 如果没有交集，返回0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个框的面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou

def process_teeth_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """处理牙齿数据"""
    # 1. 如果"Missing teeth"字段为空，则删除
    if "properties" in data and "Missing teeth" in data["properties"] and not data["properties"]["Missing teeth"]:
        del data["properties"]["Missing teeth"]
    
    # 2. 处理重复的牙齿
    if "properties" in data and "Teeth" in data["properties"] and data["properties"]["Teeth"]:
        teeth = data["properties"]["Teeth"]
        
        # 检查重叠牙齿
        to_remove = set()  # 存储需要移除的牙齿索引
        for i in range(len(teeth)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(teeth)):
                if j in to_remove:
                    continue
                    
                iou = calculate_iou(teeth[i]["bbox"], teeth[j]["bbox"])
                
                # 如果IoU > 0.95，移除score较低的牙齿
                if iou > 0.95:
                    if teeth[i]["score"] >= teeth[j]["score"]:
                        to_remove.add(j)
                        
                        # 合并条件 - 如果相同条件存在，保留score更高的
                        if "conditions" in teeth[j]:
                            for condition, details in teeth[j]["conditions"].items():
                                if condition in teeth[i].get("conditions", {}):
                                    if details.get("score", 0) > teeth[i]["conditions"][condition].get("score", 0):
                                        teeth[i]["conditions"][condition] = details
                                else:
                                    if "conditions" not in teeth[i]:
                                        teeth[i]["conditions"] = {}
                                    teeth[i]["conditions"][condition] = details
                    else:
                        to_remove.add(i)
                        
                        # 合并条件 - 如果相同条件存在，保留score更高的
                        if "conditions" in teeth[i]:
                            for condition, details in teeth[i]["conditions"].items():
                                if condition in teeth[j].get("conditions", {}):
                                    if details.get("score", 0) > teeth[j]["conditions"][condition].get("score", 0):
                                        teeth[j]["conditions"][condition] = details
                                else:
                                    if "conditions" not in teeth[j]:
                                        teeth[j]["conditions"] = {}
                                    teeth[j]["conditions"][condition] = details
                        break  # 因为i已被标记为移除，不再需要比较i和其他牙齿
        
        # 移除标记的牙齿
        data["properties"]["Teeth"] = [
            tooth for i, tooth in enumerate(teeth) if i not in to_remove
        ]
            
    return data

def process_files(input_folder: str, output_folder: str) -> None:
    """处理输入文件夹中的所有JSON文件并保存到输出文件夹"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中的所有JSON文件
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    print(f"Processing {len(json_files)} JSON files...")
    
    processed_count = 0
    error_count = 0
    
    for input_file_path in json_files:
        # 获取文件名，用于构建输出文件路径
        file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_folder, file_name)
        
        try:
            # 读取JSON文件
            with open(input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据
            processed_data = process_teeth_data(data)
            
            # 保存处理后的数据到输出文件夹
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
            processed_count += 1
            print(f"Successfully processed: {file_name}")
        
        except Exception as e:
            error_count += 1
            print(f"Error processing {file_name}: {str(e)}")
    
    print(f"Processing complete! Processed {processed_count} files, {error_count} errors.")

def main():
    """主函数，解析命令行参数并执行处理"""
    parser = argparse.ArgumentParser(description='Process dental JSON files.')
    parser.add_argument('-i', '--input', required=True, help='Input folder containing JSON files')
    parser.add_argument('-o', '--output', required=True, help='Output folder for processed JSON files')
    
    args = parser.parse_args()
    
    # 执行处理
    process_files(args.input, args.output)

if __name__ == "__main__":
    main()
