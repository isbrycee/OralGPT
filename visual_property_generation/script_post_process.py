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
    
    # 如果不能确定，默认返回
    if object_type in ['bone_loss', 'missing_teeth']:
        return "lower"  # 默认值
    else:
        return "lower left"  # 默认值

def process_json_file(file_path):
    """处理单个JSON文件"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 获取图像宽度和高度
    image_width = data.get('image_width', None)
    image_height = data.get('image_height', None)
    
    # 获取象限信息
    quadrants_info = data.get('properties', {}).get('Quadrants', [])
    
    # 1. 处理Missing teeth、MissingTeeth和Missing tooth
    missing_teeth = data.get('properties', {}).get('Missing teeth', [])
    missing_teeth_alt = data.get('properties', {}).get('MissingTeeth', [])
    missing_tooth = data.get('properties', {}).get('Missing tooth', [])
    
    # 合并Missing tooth和MissingTeeth到Missing teeth
    all_missing_teeth = []
    
    # 过滤并添加Missing teeth中score >= 0.4的
    if missing_teeth:
        all_missing_teeth.extend([mt for mt in missing_teeth if mt.get('score', 0) >= 0.4])
    
    # 过滤并添加MissingTeeth中score >= 0.4的
    if missing_teeth_alt:
        all_missing_teeth.extend([mt for mt in missing_teeth_alt if mt.get('score', 0) >= 0.4])
    
    # 过滤并添加Missing tooth中score >= 0.4的
    if missing_tooth:
        all_missing_teeth.extend([mt for mt in missing_tooth if mt.get('score', 0) >= 0.4])
    
    # 按score排序，处理重叠的缺失牙齿
    all_missing_teeth.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # 处理重叠，保留score最高的
    merged_missing_teeth = []
    for mt in all_missing_teeth:
        box1 = mt.get('bbox', [])
        # 检查是否与已合并的有重叠
        overlap = False
        for merged_mt in merged_missing_teeth:
            box2 = merged_mt.get('bbox', [])
            if calculate_iou(box1, box2) > 0.5:
                overlap = True
                break
        
        if not overlap:
            # 添加side信息
            mt['side'] = determine_side(box1, quadrants_info, 'missing_teeth', image_width, image_height)
            merged_missing_teeth.append(mt)
    
    # 更新data
    data['properties']['Missing teeth'] = merged_missing_teeth
    
    # 删除原始字段
    if 'MissingTeeth' in data.get('properties', {}):
        del data['properties']['MissingTeeth']
    if 'Missing tooth' in data.get('properties', {}):
        del data['properties']['Missing tooth']
    
    # 2. 处理tooth_id为unknown的牙齿的condition
    if 'Teeth' in data.get('properties', {}):
        teeth = data.get('properties', {}).get('Teeth', [])
        
        # 找出所有tooth_id为unknown且有conditions的牙齿
        unknown_teeth = []
        known_teeth = []
        
        for tooth in teeth:
            if tooth.get('tooth_id') == 'unknown' and 'conditions' in tooth and tooth['conditions']:
                # 过滤掉score < 0.4的conditions
                filtered_conditions = {}
                for cond_name, cond_data in tooth['conditions'].items():
                    if isinstance(cond_data.get('score', 0), list):
                        if any(score >= 0.4 for score in cond_data['score']):
                            filtered_conditions[cond_name] = cond_data
                    elif cond_data.get('score', 0) >= 0.4:
                        filtered_conditions[cond_name] = cond_data
                
                if filtered_conditions:
                    tooth_copy = tooth.copy()
                    tooth_copy['conditions'] = filtered_conditions
                    unknown_teeth.append(tooth_copy)
            else:
                known_teeth.append(tooth)
        
        # 处理unknown牙齿的conditions，将它们分配给重叠的已知牙齿
        for unknown_tooth in unknown_teeth:
            for cond_name, cond_data in unknown_tooth['conditions'].items():
                # 检查condition是否有bbox
                if 'bbox' in cond_data:
                    # 对于bbox列表的情况
                    if isinstance(cond_data['bbox'], list) and isinstance(cond_data['bbox'][0], list):
                        for i, bbox in enumerate(cond_data['bbox']):
                            # 查找与该condition bbox重叠最大的牙齿
                            max_iou = 0
                            best_tooth = None
                            
                            for known_tooth in known_teeth:
                                if 'bbox' in known_tooth:
                                    known_bbox = known_tooth['bbox']
                                    iou = calculate_iou(bbox, known_bbox)
                                    
                                    if iou > max_iou:
                                        max_iou = iou
                                        best_tooth = known_tooth
                            
                            # 如果找到重叠的牙齿，将condition添加到该牙齿中
                            if best_tooth and max_iou > 0:  # 只要有任何重叠即可分配
                                if 'conditions' not in best_tooth:
                                    best_tooth['conditions'] = {}
                                
                                # 如果condition已存在于目标牙齿中，比较score
                                if cond_name in best_tooth['conditions']:
                                    existing_cond = best_tooth['conditions'][cond_name]
                                    existing_score = existing_cond.get('score', 0)
                                    
                                    # 获取当前condition的score
                                    if isinstance(cond_data.get('score', 0), list):
                                        new_score = cond_data['score'][i] if i < len(cond_data['score']) else 0
                                    else:
                                        new_score = cond_data.get('score', 0)
                                    
                                    # 如果新的score更高，替换condition
                                    if new_score > existing_score:
                                        # 创建新的condition数据
                                        new_cond = {
                                            'present': cond_data.get('present', True),
                                            'bbox': bbox,
                                            'score': new_score
                                        }
                                        
                                        # 如果有segmentation，也复制
                                        if 'segmentation' in cond_data and isinstance(cond_data['segmentation'], list):
                                            if i < len(cond_data['segmentation']):
                                                new_cond['segmentation'] = cond_data['segmentation'][i]
                                        
                                        best_tooth['conditions'][cond_name] = new_cond
                                else:
                                    # 如果condition不存在，直接添加
                                    # 创建新的condition数据
                                    new_cond = {
                                        'present': cond_data.get('present', True),
                                        'bbox': bbox,
                                        'score': cond_data['score'][i] if isinstance(cond_data.get('score', 0), list) and i < len(cond_data['score']) else cond_data.get('score', 0)
                                    }
                                    
                                    # 如果有segmentation，也复制
                                    if 'segmentation' in cond_data and isinstance(cond_data['segmentation'], list):
                                        if i < len(cond_data['segmentation']):
                                            new_cond['segmentation'] = cond_data['segmentation'][i]
                                    
                                    best_tooth['conditions'][cond_name] = new_cond
                    else:
                        # 对于单个bbox的情况
                        bbox = cond_data['bbox']
                        
                        # 查找与该condition bbox重叠最大的牙齿
                        max_iou = 0
                        best_tooth = None
                        
                        for known_tooth in known_teeth:
                            if 'bbox' in known_tooth:
                                known_bbox = known_tooth['bbox']
                                iou = calculate_iou(bbox, known_bbox)
                                
                                if iou > max_iou:
                                    max_iou = iou
                                    best_tooth = known_tooth
                        
                        # 如果找到重叠的牙齿，将condition添加到该牙齿中
                        if best_tooth and max_iou > 0:  # 只要有任何重叠即可分配
                            if 'conditions' not in best_tooth:
                                best_tooth['conditions'] = {}
                            
                            # 如果condition已存在于目标牙齿中，比较score
                            if cond_name in best_tooth['conditions']:
                                existing_cond = best_tooth['conditions'][cond_name]
                                existing_score = existing_cond.get('score', 0)
                                new_score = cond_data.get('score', 0)
                                
                                # 如果新的score更高，替换condition
                                if new_score > existing_score:
                                    best_tooth['conditions'][cond_name] = cond_data.copy()
                            else:
                                # 如果condition不存在，直接添加
                                best_tooth['conditions'][cond_name] = cond_data.copy()
        
        # 更新Teeth字段，移除tooth_id为unknown的牙齿
        data['properties']['Teeth'] = [tooth for tooth in teeth if tooth.get('tooth_id') != 'unknown']
    
    # 3. 处理JawBones，过滤score低于0.4的所有条件
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
