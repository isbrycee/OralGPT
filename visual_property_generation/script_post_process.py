import os
import json
import numpy as np
from glob import glob

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    # box格式为[x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算相交矩形的坐标
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # 没有相交区域
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算相交区域面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算两个边界框的面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IOU
    iou = intersection_area / union_area
    
    return iou

def calculate_box_area(box):
    """计算边界框的面积"""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def calculate_overlap_area(box1, box2):
    """计算两个边界框的重叠面积"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算相交矩形的坐标
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # 没有相交区域
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算相交区域面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def determine_side(box, quadrants_info, object_type='tooth', image_width=None, image_height=None):
    """
    根据box和象限位置确定side
    
    参数:
    - box: 边界框坐标 [x1, y1, x2, y2]
    - quadrants_info: 象限信息列表
    - object_type: 对象类型，'tooth'、'bone_loss' 或 'missing_teeth'
    - image_width: 图像宽度
    - image_height: 图像高度
    
    返回:
    - side: 确定的side值
    """
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    
    # 计算box与每个象限的重叠面积
    overlaps = []
    for q in quadrants_info:
        q_box = q['bbox']
        overlap_area = calculate_overlap_area(box, q_box)
        if overlap_area > 0:
            overlaps.append((q['quadrant'], overlap_area))
    
    # 按重叠面积排序
    overlaps.sort(key=lambda x: x[1], reverse=True)
    
    # 如果有重叠，选择重叠面积最大的象限
    if overlaps:
        best_quadrant = overlaps[0][0]
        
        # 对于bone_loss和missing_teeth可以使用upper或lower
        if object_type in ['bone_loss', 'missing_teeth']:
            upper_area = sum(area for q, area in overlaps if "1" in q or "2" in q)
            lower_area = sum(area for q, area in overlaps if "3" in q or "4" in q)
            
            if upper_area > 0 and lower_area == 0:
                return "upper"
            elif lower_area > 0 and upper_area == 0:
                return "lower"
            elif upper_area > lower_area:
                return "upper"
            elif lower_area > upper_area:
                return "lower"
            else:
                # 如果上下象限重叠面积相等，使用中心点判断
                if image_height is not None and y_center < image_height / 2:
                    return "upper"
                else:
                    return "lower"
        
        # 根据最佳象限确定side
        if "1" in best_quadrant:
            return "upper right"
        elif "2" in best_quadrant:
            return "upper left"
        elif "3" in best_quadrant:
            return "lower left"
        elif "4" in best_quadrant:
            return "lower right"
    
    # 如果没有和任何象限重叠，使用中心点判断
    if image_height is not None and image_width is not None:
        midpoint_y = image_height / 2
        midpoint_x = image_width / 2
        
        if y_center < midpoint_y:
            if object_type in ['bone_loss', 'missing_teeth']:
                return "upper"
            else:
                # 根据水平位置确定左右
                if x_center < midpoint_x:
                    return "upper right"
                else:
                    return "upper left"
        else:
            if object_type in ['bone_loss', 'missing_teeth']:
                return "lower"
            else:
                # 根据水平位置确定左右
                if x_center < midpoint_x:
                    return "lower right"
                else:
                    return "lower left"
    

def process_json_file(file_path):
    """处理单个JSON文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 获取图像宽度和高度
    image_width = data.get('image_width', None)
    image_height = data.get('image_height', None)
    
    # 获取象限信息
    quadrants_info = data.get('properties', {}).get('Quadrants', [])
    
    # 1. 处理MissingTeeth和Missing teeth
    missing_teeth = data.get('properties', {}).get('Missing teeth', [])
    missing_teeth_alt = data.get('properties', {}).get('MissingTeeth', [])
    
    # 过滤Missing teeth中score低于0.4的
    if missing_teeth:
        missing_teeth = [mt for mt in missing_teeth if mt.get('score', 0) >= 0.4]
    
    # 过滤MissingTeeth中score低于0.4的
    if missing_teeth_alt:
        missing_teeth_alt = [mt for mt in missing_teeth_alt if mt.get('score', 0) >= 0.4]
    
    # 如果两个字段都存在
    if missing_teeth and missing_teeth_alt:
        merged_teeth = []
        
        # 检查重叠并选择score高的
        for mt1 in missing_teeth:
            added = False
            box1 = mt1.get('bbox', [])
            
            for mt2 in missing_teeth_alt:
                box2 = mt2.get('bbox', [])
                
                if calculate_iou(box1, box2) > 0.5:  # 重叠阈值可以调整
                    # 选择score更高的
                    if mt2.get('score', 0) > mt1.get('score', 0):
                        mt2_copy = mt2.copy()
                        # 添加side信息
                        mt2_copy['side'] = determine_side(box2, quadrants_info, 'missing_teeth', image_width, image_height)
                        merged_teeth.append(mt2_copy)
                    else:
                        mt1_copy = mt1.copy()
                        # 添加side信息
                        mt1_copy['side'] = determine_side(box1, quadrants_info, 'missing_teeth', image_width, image_height)
                        merged_teeth.append(mt1_copy)
                    added = True
                    break
            
            if not added:
                mt1_copy = mt1.copy()
                # 添加side信息
                mt1_copy['side'] = determine_side(box1, quadrants_info, 'missing_teeth', image_width, image_height)
                merged_teeth.append(mt1_copy)
        
        # 添加未重叠的missing_teeth_alt
        for mt2 in missing_teeth_alt:
            box2 = mt2.get('bbox', [])
            
            # 检查是否已经添加过
            already_added = False
            for merged_mt in merged_teeth:
                if np.array_equal(merged_mt.get('bbox', []), box2):
                    already_added = True
                    break
            
            if not already_added:
                mt2_copy = mt2.copy()
                # 添加side信息
                mt2_copy['side'] = determine_side(box2, quadrants_info, 'missing_teeth', image_width, image_height)
                merged_teeth.append(mt2_copy)
        
        # 更新data
        data['properties']['Missing teeth'] = merged_teeth
        if 'MissingTeeth' in data.get('properties', {}):
            del data['properties']['MissingTeeth']
    
    # 如果只有MissingTeeth
    elif missing_teeth_alt and not missing_teeth:
        for mt in missing_teeth_alt:
            # 添加side信息
            mt['side'] = determine_side(mt.get('bbox', []), quadrants_info, 'missing_teeth', image_width, image_height)
        
        data['properties']['Missing teeth'] = missing_teeth_alt
        del data['properties']['MissingTeeth']
    
    # 如果只有Missing teeth，添加side信息
    elif missing_teeth and not missing_teeth_alt:
        for mt in missing_teeth:
            # 添加side信息
            mt['side'] = determine_side(mt.get('bbox', []), quadrants_info, 'missing_teeth', image_width, image_height)
        
        data['properties']['Missing teeth'] = missing_teeth
    
    # 处理JawBones，过滤score低于0.4的所有条件
    if 'JawBones' in data.get('properties', {}):
        for jawbone in data['properties']['JawBones']:
            if 'conditions' in jawbone:
                conditions_to_remove = []
                
                for condition_name, condition_data in jawbone['conditions'].items():
                    # 处理bone loss，只保留score > 0.6的
                    if condition_name == 'Bone loss':
                        if isinstance(condition_data.get('score', []), list):
                            filtered_boxes = []
                            filtered_segmentations = []
                            filtered_scores = []
                            
                            for i, score in enumerate(condition_data['score']):
                                if score > 0.6:  # 对bone loss使用0.6的阈值
                                    filtered_boxes.append(condition_data['bbox'][i])
                                    filtered_segmentations.append(condition_data['segmentation'][i])
                                    filtered_scores.append(score)
                            
                            if filtered_boxes:
                                condition_data['bbox'] = filtered_boxes
                                condition_data['segmentation'] = filtered_segmentations
                                condition_data['score'] = filtered_scores
                                
                                # 添加side信息
                                sides = []
                                for box in filtered_boxes:
                                    sides.append(determine_side(box, quadrants_info, 'bone_loss', image_width, image_height))
                                condition_data['side'] = sides
                            else:
                                # 如果没有满足条件的bone loss，标记为删除
                                conditions_to_remove.append(condition_name)
                        else:
                            # 对于单个score的情况
                            if condition_data.get('score', 0) <= 0.6:  # 对bone loss使用0.6的阈值
                                conditions_to_remove.append(condition_name)
                            else:
                                # 添加side信息
                                condition_data['side'] = determine_side(condition_data['bbox'], quadrants_info, 'bone_loss', image_width, image_height)
                    
                    # 处理其他条件，过滤score < 0.4的
                    else:
                        if isinstance(condition_data.get('score', []), list):
                            filtered_boxes = []
                            filtered_segmentations = []
                            filtered_scores = []
                            
                            for i, score in enumerate(condition_data['score']):
                                if score >= 0.4:
                                    if 'bbox' in condition_data:
                                        filtered_boxes.append(condition_data['bbox'][i])
                                    if 'segmentation' in condition_data:
                                        filtered_segmentations.append(condition_data['segmentation'][i])
                                    filtered_scores.append(score)
                            
                            if filtered_scores:
                                if filtered_boxes:
                                    condition_data['bbox'] = filtered_boxes
                                if filtered_segmentations:
                                    condition_data['segmentation'] = filtered_segmentations
                                condition_data['score'] = filtered_scores
                            else:
                                # 如果没有满足条件的项，标记为删除
                                conditions_to_remove.append(condition_name)
                        else:
                            # 对于单个score的情况
                            if condition_data.get('score', 0) < 0.4:
                                conditions_to_remove.append(condition_name)
                
                # 删除被标记的条件
                for condition_name in conditions_to_remove:
                    del jawbone['conditions'][condition_name]
    
    return data

def process_folder(folder_path, output_folder=None):
    """处理文件夹中的所有JSON文件"""
    if output_folder is None:
        output_folder = os.path.join(folder_path, 'processed')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob(os.path.join(folder_path, '*.json'))
    
    for file_path in json_files:
        try:
            # 处理文件
            processed_data = process_json_file(file_path)
            
            # 保存处理后的文件
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_folder, file_name)
            
            with open(output_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"Processed: {file_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='caption json post process')
    parser.add_argument('--input', required=True, help='输入json目录')
    parser.add_argument('--output', required=True, help='输出json目录')
    
    args = parser.parse_args()
    folder_path = args.input
    output_path = args.output
    
    process_folder(folder_path, output_path)
