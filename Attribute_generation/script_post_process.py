import json
import os
import glob
import argparse
import copy
from typing import Dict, List, Any, Union, Tuple

def calculate_iou(box1: List[Union[int, float]], box2: List[Union[int, float]]) -> float:
    """计算两个边界框的交并比(IoU)"""
    # 处理空边界框
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    
    # 确保使用[x1, y1, x2, y2]格式
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 检查是否需要转换格式
    if x2_1 < x1_1 and y2_1 < y1_1:  # 可能是[x2, y2, x1, y1]格式
        x1_1, y1_1, x2_1, y2_1 = x2_1, y2_1, x1_1, y1_1
    elif x2_1 > 0 and y2_1 > 0 and not (x2_1 < x1_1 and y2_1 < y1_1):  # 如果是[x, y, w, h]格式
        x2_1, y2_1 = x1_1 + x2_1, y1_1 + y2_1
    
    if x2_2 < x1_2 and y2_2 < y1_2:  # 可能是[x2, y2, x1, y1]格式
        x1_2, y1_2, x2_2, y2_2 = x2_2, y2_2, x1_2, y1_2
    elif x2_2 > 0 and y2_2 > 0 and not (x2_2 < x1_2 and y2_2 < y1_2):  # 如果是[x, y, w, h]格式
        x2_2, y2_2 = x1_2 + x2_2, y1_2 + y2_2
    
    # 计算交集区域
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集区域
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def get_highest_score(score_value):
    """从分数值中获取最高分数，处理列表和单一值"""
    if isinstance(score_value, list):
        if not score_value:  # 如果是空列表
            return 0.0
        return max(filter(lambda x: isinstance(x, (int, float)), score_value), default=0.0)
    elif isinstance(score_value, (int, float)):
        return score_value
    return 0.0

def determine_side_from_bbox(bbox: List[Union[int, float]], 
                             quadrants: Dict[str, List[Union[int, float]]], 
                             object_type: str = 'tooth',
                             image_width: int = None, 
                             image_height: int = None) -> Union[str, List[str]]:
    """
    根据边界框确定其所在的象限和侧面
    
    参数:
    - bbox: 单个边界框或边界框列表
    - quadrants: 象限信息字典
    - object_type: 对象类型，'tooth'、'bone_loss'或'missing_teeth'
    - image_width: 图像宽度
    - image_height: 图像高度
    
    返回:
    - 确定的side值(字符串)或side值列表
    """
    # 处理bbox为列表的列表的情况
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
        sides = []
        for single_bbox in bbox:
            side = determine_side_from_bbox(single_bbox, quadrants, object_type, image_width, image_height)
            sides.append(side)
        return sides
    
    # 处理单个bbox的情况
    if not bbox:  # 如果边界框为空
        return "unknown"
    
    # 计算边界框中心点
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
        
    best_overlap = 0
    best_quadrant = None
    
    for quadrant_name, quadrant_bbox in quadrants.items():
        overlap = calculate_iou(bbox, quadrant_bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best_quadrant = quadrant_name
    
    # 对于bone_loss和missing_teeth可以使用upper或lower
    if object_type in ['bone_loss', 'missing_teeth']:
        if best_quadrant:
            if "1" in best_quadrant or "2" in best_quadrant:
                return "upper"
            elif "3" in best_quadrant or "4" in best_quadrant:
                return "lower"
        
        # 如果没有找到最佳象限，使用中心点的y坐标
        if image_height:
            return "upper" if y_center < image_height / 2 else "lower"
    
    # 对于牙齿，返回具体的象限
    if best_quadrant:
        if "1" in best_quadrant:
            return "upper right"
        elif "2" in best_quadrant:
            return "upper left"
        elif "3" in best_quadrant:
            return "lower left"
        elif "4" in best_quadrant:
            return "lower right"
    
    # 如果无法通过象限确定，使用中心点坐标
    if image_height and image_width:
        if y_center < image_height / 2:  # 上半部分
            return "upper right" if x_center < image_width / 2 else "upper left"
        else:  # 下半部分
            return "lower left" if x_center > image_width / 2 else "lower right"
    
    return "unknown"

def ensure_bbox_list(bbox_data) -> List[List[Union[int, float]]]:
    """
    确保bbox_data是格式为[[x1, y1, x2, y2], ...]的列表。
    如果是单个bbox [x1, y1, x2, y2]，转换为[[x1, y1, x2, y2]]。
    """
    if isinstance(bbox_data, list):
        if bbox_data and isinstance(bbox_data[0], list):
            return [box for box in bbox_data if len(box) == 4 and all(isinstance(x, (int, float)) for x in box)]
        elif len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
            return [bbox_data]
    return []

def is_valid_bbox(bbox: Any) -> bool:
    """检查边界框是否有效"""
    if not isinstance(bbox, list):
        return False
    
    if not bbox:  # 空列表
        return False
    
    # 处理嵌套列表的情况 [[x1, y1, x2, y2]]
    if isinstance(bbox[0], list):
        return False
    
    # 检查是否有4个值
    if len(bbox) != 4:
        return False
    
    return True

def process_dental_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据指定要求处理牙科JSON数据
    
    处理内容包括:
    1. 删除空的"Missing teeth"字段
    2. 合并和处理各种Missing teeth相关字段
    3. 处理牙齿数据，包括处理重复牙齿
    4. 过滤低分条件
    5. 处理unknown牙齿的条件
    6. 处理根尖和残根条件
    7. 为特定牙齿添加推荐提取
    8. 处理Crown和Filling的重叠
    """
    result = copy.deepcopy(data)
    
    # 获取图像尺寸
    image_width = result.get('image_width', None)
    image_height = result.get('image_height', None)
    
    # 获取象限信息
    quadrants = {}
    if "properties" in result and "Quadrants" in result["properties"]:
        for quadrant_info in result["properties"]["Quadrants"]:
            quadrant_name = quadrant_info["quadrant"]
            quadrant_bbox = quadrant_info["bbox"]
            quadrants[quadrant_name] = quadrant_bbox
    
    # 1. 如果"Missing teeth"字段为空，则删除
    if "properties" in result and "Missing teeth" in result["properties"] and not result["properties"]["Missing teeth"]:
        del result["properties"]["Missing teeth"]
    
    # 2. 合并和处理各种Missing teeth相关字段
    if "properties" in result:
        missing_teeth = result["properties"].get("Missing teeth", [])
        missing_teeth_alt = result["properties"].get("MissingTeeth", [])
        missing_tooth = result["properties"].get("Missing tooth", [])
        
        # 合并Missing tooth和MissingTeeth到Missing teeth
        all_missing_teeth = []
        
        # 过滤并添加Missing teeth中score >= 0.4的
        if missing_teeth:
            all_missing_teeth.extend([mt for mt in missing_teeth if get_highest_score(mt.get('score', 0)) >= 0.4])
        
        # 过滤并添加MissingTeeth中score >= 0.4的
        if missing_teeth_alt:
            all_missing_teeth.extend([mt for mt in missing_teeth_alt if get_highest_score(mt.get('score', 0)) >= 0.4])
        
        # 过滤并添加Missing tooth中score >= 0.4的
        if missing_tooth:
            all_missing_teeth.extend([mt for mt in missing_tooth if get_highest_score(mt.get('score', 0)) >= 0.4])
        
        # 按score排序，处理重叠的缺失牙齿
        all_missing_teeth.sort(key=lambda x: get_highest_score(x.get('score', 0)), reverse=True)
        
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
                mt['side'] = determine_side_from_bbox(box1, quadrants, 'missing_teeth', image_width, image_height)
                merged_missing_teeth.append(mt)
        
        # 更新data
        if merged_missing_teeth:
            result["properties"]["Missing teeth"] = merged_missing_teeth
        elif "Missing teeth" in result["properties"]:
            del result["properties"]["Missing teeth"]
        
        # 删除原始字段
        if 'MissingTeeth' in result["properties"]:
            del result["properties"]["MissingTeeth"]
        if 'Missing tooth' in result["properties"]:
            del result["properties"]["Missing tooth"]
    
    # 3. 处理牙齿数据，包括处理重复牙齿
    if "properties" in result and "Teeth" in result["properties"]:
        teeth = result["properties"]["Teeth"]
        teeth_dict = {}
        unknown_teeth = []
        
        # 3.1 分离known和unknown牙齿，处理重复牙号
        for tooth in teeth:
            tooth_id = tooth.get("tooth_id", "unknown")
            score = tooth.get("score", 0.0)
            
            # 获取分数，处理可能是列表的情况
            tooth_score = get_highest_score(score)
            
            # 将unknown牙齿单独处理
            if tooth_id == "unknown":
                unknown_teeth.append(tooth)
                continue
            
            if tooth_id in teeth_dict:
                existing_score = get_highest_score(teeth_dict[tooth_id].get("score", 0.0))
                if tooth_score > existing_score:
                    teeth_dict[tooth_id] = tooth
            else:
                teeth_dict[tooth_id] = tooth
        
        # 3.2 过滤掉所有score<0.4的所有condition/disease
        for tooth in list(teeth_dict.values()) + unknown_teeth:
            if "conditions" in tooth:
                conditions_to_keep = {}
                for condition_name, condition_data in tooth["conditions"].items():
                    # 特殊处理Periapical lesions和Impacted tooth
                    if condition_name == "Periapical lesions":
                        threshold = 0.7
                    elif condition_name == "Impacted tooth":
                        threshold = 0.55
                    else:
                        threshold = 0.4
                    
                    # 获取分数
                    condition_score = get_highest_score(condition_data.get("score", 0))
                    
                    # 保留高于阈值的条件
                    if condition_score >= threshold:
                        # 添加side信息给unknown牙齿的条件
                        if tooth.get("tooth_id") == "unknown" and "bbox" in condition_data:
                            side = determine_side_from_bbox(
                                condition_data["bbox"], 
                                quadrants, 
                                'tooth', 
                                image_width, 
                                image_height
                            )
                            condition_data["side"] = side
                        
                        conditions_to_keep[condition_name] = condition_data
                
                tooth["conditions"] = conditions_to_keep
        
        # 3.3 处理根尖和残根条件
        for tooth in list(teeth_dict.values()) + unknown_teeth:
            if "conditions" in tooth:
                has_root_piece = "Root piece" in tooth["conditions"]
                has_retained_root = "Retained root" in tooth["conditions"]
                
                if has_root_piece and not has_retained_root:
                    # 将Root piece转换为Retained root
                    tooth["conditions"]["Retained root"] = tooth["conditions"]["Root piece"]
                    del tooth["conditions"]["Root piece"]
                elif has_root_piece and has_retained_root:
                    # 只保留Retained root
                    del tooth["conditions"]["Root piece"]
        
        # 3.4 为特定条件的智齿添加"Recommended extraction"
        has_18 = "18" in teeth_dict
        has_28 = "28" in teeth_dict
        has_38 = "38" in teeth_dict
        has_48 = "48" in teeth_dict
        
        if has_18 and not has_48:
            if "conditions" not in teeth_dict["18"]:
                teeth_dict["18"]["conditions"] = {}
            teeth_dict["18"]["conditions"]["Recommended extraction"] = {"present": True}
        
        if has_28 and not has_38:
            if "conditions" not in teeth_dict["28"]:
                teeth_dict["28"]["conditions"] = {}
            teeth_dict["28"]["conditions"]["Recommended extraction"] = {"present": True}
        
        # 3.5 处理unknown牙齿条件的分配
        unknown_teeth_to_keep = []
        for unknown_tooth in unknown_teeth:
            if "conditions" not in unknown_tooth or not unknown_tooth["conditions"]:
                continue
                
            all_conditions_assigned = True
            conditions_to_remove = []
            
            for cond_name, cond_data in unknown_tooth["conditions"].items():
                if "bbox" in cond_data:
                    # 确保bbox为列表格式
                    bbox_list = ensure_bbox_list(cond_data["bbox"])
                    
                    if bbox_list:
                        # 遍历每个bbox，尝试找到重叠的known牙齿
                        for bbox in bbox_list:
                            max_iou = 0
                            best_tooth = None
                            
                            for tooth_id, known_tooth in teeth_dict.items():
                                if "bbox" in known_tooth and is_valid_bbox(known_tooth["bbox"]):
                                    iou = calculate_iou(bbox, known_tooth["bbox"])
                                    if iou > max_iou:
                                        max_iou = iou
                                        best_tooth = known_tooth
                            
                            # 如果找到重叠的牙齿，添加条件
                            if best_tooth and max_iou > 0.5:
                                if "conditions" not in best_tooth:
                                    best_tooth["conditions"] = {}
                                
                                if cond_name not in best_tooth["conditions"]:
                                    best_tooth["conditions"][cond_name] = cond_data.copy()
                                elif get_highest_score(cond_data.get("score", 0)) > get_highest_score(best_tooth["conditions"][cond_name].get("score", 0)):
                                    best_tooth["conditions"][cond_name] = cond_data.copy()
                                
                                conditions_to_remove.append(cond_name)
                            else:
                                all_conditions_assigned = False
            
            # 从unknown_tooth移除已分配的条件
            for cond_name in conditions_to_remove:
                if cond_name in unknown_tooth["conditions"]:
                    del unknown_tooth["conditions"][cond_name]
            
            # 如果还有未分配的条件，保留这个unknown牙齿
            if not all_conditions_assigned and unknown_tooth["conditions"]:
                unknown_teeth_to_keep.append(unknown_tooth)
        
        # 3.6 处理Crown和Filling的重叠
        for tooth in list(teeth_dict.values()) + unknown_teeth_to_keep:
            if "conditions" in tooth and "Crown" in tooth["conditions"] and "Filling" in tooth["conditions"]:
                crown = tooth["conditions"]["Crown"]
                filling = tooth["conditions"]["Filling"]
                
                if "bbox" in crown and "bbox" in filling:
                    crown_bboxes = ensure_bbox_list(crown["bbox"])
                    filling_bboxes = ensure_bbox_list(filling["bbox"])
                    
                    if crown_bboxes and filling_bboxes:
                        crown_scores = crown.get("score", 0)
                        if not isinstance(crown_scores, list):
                            crown_scores = [crown_scores] * len(crown_bboxes)
                        
                        filling_scores = filling.get("score", 0)
                        if not isinstance(filling_scores, list):
                            filling_scores = [filling_scores] * len(filling_bboxes)
                        
                        # 检查每对Crown和Filling的重叠
                        for i, (crown_bbox, crown_score) in enumerate(zip(crown_bboxes, crown_scores)):
                            for j, (filling_bbox, filling_score) in enumerate(zip(filling_bboxes, filling_scores)):
                                iou = calculate_iou(crown_bbox, filling_bbox)
                                
                                # 如果IoU很高，保留得分更高的
                                if iou > 0.9:
                                    if crown_score < filling_score:
                                        # 移除这个crown
                                        if isinstance(crown["bbox"], list) and isinstance(crown["bbox"][0], list):
                                            crown["bbox"].pop(i)
                                            if isinstance(crown["score"], list):
                                                crown["score"].pop(i)
                                        else:
                                            del tooth["conditions"]["Crown"]
                                            break
                                    else:
                                        # 移除这个filling
                                        if isinstance(filling["bbox"], list) and isinstance(filling["bbox"][0], list):
                                            filling["bbox"].pop(j)
                                            if isinstance(filling["score"], list):
                                                filling["score"].pop(j)
                                        else:
                                            del tooth["conditions"]["Filling"]
                                            break
        
        # 更新teeth列表
        result["properties"]["Teeth"] = list(teeth_dict.values()) + unknown_teeth_to_keep
    
    # 4. 处理JawBones，过滤score低于阈值的所有条件
    if "properties" in result and "JawBones" in result["properties"]:
        for jawbone in result["properties"]["JawBones"]:
            if "conditions" in jawbone:
                conditions_to_remove = []
                
                for condition_name, condition_data in jawbone["conditions"].items():
                    # 处理bone loss，使用更高的阈值
                    if condition_name == "Bone loss":
                        threshold = 0.6
                    else:
                        threshold = 0.4
                    
                    # 处理bbox和score可能是列表的情况
                    if isinstance(condition_data.get("score", []), list):
                        filtered_boxes = []
                        filtered_segmentations = []
                        filtered_scores = []
                        
                        for i, score in enumerate(condition_data["score"]):
                            if score > threshold:
                                if "bbox" in condition_data and i < len(condition_data["bbox"]):
                                    filtered_boxes.append(condition_data["bbox"][i])
                                if "segmentation" in condition_data and i < len(condition_data["segmentation"]):
                                    filtered_segmentations.append(condition_data["segmentation"][i])
                                filtered_scores.append(score)
                        
                        if filtered_boxes:
                            condition_data["bbox"] = filtered_boxes
                            condition_data["score"] = filtered_scores
                            if filtered_segmentations:
                                condition_data["segmentation"] = filtered_segmentations
                            
                            # 添加side信息
                            sides = []
                            for box in filtered_boxes:
                                sides.append(determine_side_from_bbox(box, quadrants, 'bone_loss', image_width, image_height))
                            condition_data["side"] = sides
                        else:
                            conditions_to_remove.append(condition_name)
                    else:
                        # 单个score的情况
                        if condition_data.get("score", 0) < threshold:
                            conditions_to_remove.append(condition_name)
                        elif "bbox" in condition_data:
                            # 添加side信息
                            condition_data["side"] = determine_side_from_bbox(
                                condition_data["bbox"], 
                                quadrants, 
                                'bone_loss', 
                                image_width, 
                                image_height
                            )
                
                # 删除被标记的条件
                for condition_name in conditions_to_remove:
                    del jawbone["conditions"][condition_name]
    
    return result

def process_directory(input_dir: str, output_dir: str) -> None:
    """处理目录中的所有JSON文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有json文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    print(f"找到 {len(json_files)} 个JSON文件待处理")
    
    successful = 0
    failed = 0
    
    for input_file in json_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        try:
            # 读取JSON文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据
            processed_data = process_dental_json(data)
            
            # 保存处理后的数据
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            successful += 1
            print(f"已处理: {filename}")
            
        except Exception as e:
            failed += 1
            print(f"处理 {filename} 时出错: {str(e)}")
    
    print(f"处理完成! 成功: {successful}, 失败: {failed}")

def main():
    """主函数，解析命令行参数并执行处理"""
    parser = argparse.ArgumentParser(description='后处理牙科JSON数据')
    parser.add_argument('-i', '--input', required=True, help='输入JSON文件或目录')
    parser.add_argument('-o', '--output', required=True, help='输出JSON文件或目录')
    
    args = parser.parse_args()
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        try:
            # 读取JSON文件
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据
            processed_data = process_dental_json(data)
            
            # 保存处理后的数据
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
            print(f"已成功处理文件: {args.input} -> {args.output}")
        
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
    
    elif os.path.isdir(args.input):
        # 处理目录
        process_directory(args.input, args.output)
    
    else:
        print(f"错误: 输入路径 '{args.input}' 不存在")

if __name__ == "__main__":
    main()
