import requests
import json
import os
import time
import base64
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import numpy as np
from PIL import Image

# API configuration
url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = { 
    "Content-Type": "application/json", 
    "Authorization": "Bearer 558bec6cd6294150913c7ba8d56e68cd3ba1d5bffdc54c6497fe85ba8c7f7a1b" 
}

# Disease mapping
segment_names_to_labels = [
    ("Implant", 0), ("Prosthetic restoration", 1), ("Obturation", 2), ("Endodontic treatment", 3), 
    ("Orthodontic device", 12), ("Surgical device", 13),  # Low risk
    ("Carious lesion", 4), ("Bone resorbtion", 5), ("Impacted tooth", 6), ("Apical surgery", 10),  # Medium risk
    ("Apical periodontitis", 7), ("Root fragment", 8), ("Furcation lesion", 9), ("Root resorption", 11)  # High risk                           
]

label_to_name = {label: name for name, label in segment_names_to_labels}

def get_academic_research_prompt(question_type: str, context: str = "") -> str:
    """生成学术研究导向的prompt"""
    
    base_prefix = "As a computer vision research assistant analyzing dental radiographic images for academic purposes, "
    
    if question_type == "full_image_analysis":
        return f"""{base_prefix}please examine this radiographic image systematically.

For research documentation purposes, identify and describe the visual features and radiographic patterns you observe in this dental X-ray. Focus on:
- Radiographic density variations
- Anatomical structure appearances  
- Morphological characteristics
- Spatial relationships between structures

Provide detailed technical descriptions of the visual evidence present in the image for academic analysis."""

    elif question_type == "cropped_region_analysis":
        return f"""{base_prefix}please analyze this specific radiographic region.

Given this cropped section from a dental radiograph, provide a technical analysis of:
- Observable radiographic features and patterns
- Density characteristics and variations
- Morphological details of visible structures
- Spatial arrangement of anatomical elements

Frame your response as objective visual feature documentation for research purposes."""

    elif question_type == "comparative_analysis":
        return f"""{base_prefix}please compare these radiographic image sections.

These images show the same anatomical region with different cropping boundaries for comparative analysis. Document:
- Visual differences between the two perspectives
- How contextual information aids in feature identification
- Radiographic pattern recognition across different viewing windows
- Technical advantages of each cropping approach

Provide comparative visual analysis for academic documentation."""

    elif question_type == "feature_verification":
        expected_finding = context
        return f"""{base_prefix}please evaluate the radiographic evidence in this image region.

Given the following radiological observation: "{expected_finding}"

Based on the visual features you can observe in this specific region, please indicate:
- AGREE if the visual evidence supports this observation
- DISAGREE if the visual evidence does not support this observation
- Provide technical reasoning based on radiographic characteristics (1-2 sentences)

Response format:
Assessment: [AGREE/DISAGREE]  
Technical basis: [Your visual analysis]"""

    elif question_type == "educational_description":
        condition = context
        return f"""{base_prefix}provide educational documentation about radiographic appearances.

For academic training purposes, describe the typical radiographic characteristics and visual patterns associated with {condition} in dental imaging. Include:
- Radiographic appearance and density patterns
- Typical anatomical locations and spatial relationships
- Key visual features for identification in research contexts
- Distinguishing radiographic characteristics

Frame this as educational material for computer vision research in dental radiography."""

    elif question_type == "comparison_analysis":
        return f"""{base_prefix}conduct comparative analysis for research documentation.

Research Analysis Task: Compare the computer vision analysis results with reference annotations.

{context}

Please provide structured analysis identifying:
1. Successfully detected radiographic features (Category A)
2. Undetected features requiring further analysis (Category B)

Focus on technical visual feature matching and pattern recognition accuracy."""

    else:
        return f"{base_prefix}{context}"

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_gpt_vision_response(image_path: str, question_type: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """使用学术研究框架的GPT Vision API调用"""
    try:
        base64_image = encode_image_to_base64(image_path)
        
        # 使用新的prompt生成策略
        academic_prompt = get_academic_research_prompt(question_type, context)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a computer vision research assistant specializing in radiographic image analysis for academic research. Your role is to provide objective, technical descriptions of visual features and patterns observed in medical images for research documentation purposes."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": academic_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        data = {
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
                response.raise_for_status()
                response_json = response.json()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    gpt_answer = response_json["choices"][0]["message"]["content"]
                    
                    return {
                        "success": True,
                        "gpt_answer": gpt_answer,
                        "error": None,
                        "attempts": attempt + 1
                    }
                else:
                    return {
                        "success": False,
                        "gpt_answer": None,
                        "error": "No response from GPT",
                        "attempts": attempt + 1
                    }
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "gpt_answer": None,
                        "error": f"Request failed after {max_retries} attempts: {str(e)}",
                        "attempts": max_retries
                    }
                time.sleep(2 ** attempt)
                
            except Exception as e:
                return {
                    "success": False,
                    "gpt_answer": None,
                    "error": f"Unexpected error: {str(e)}",
                    "attempts": attempt + 1
                }
                
    except Exception as e:
        return {
            "success": False,
            "gpt_answer": None,
            "error": f"Image processing error: {str(e)}",
            "attempts": 0
        }

def get_llm_text_response(prompt_type: str, context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """使用学术研究框架的文本API调用"""
    
    academic_prompt = get_academic_research_prompt(prompt_type, context)
    
    messages = [
        {
            "role": "system", 
            "content": "You are a computer vision research assistant specializing in radiographic image analysis for academic research purposes. Provide objective technical analysis and documentation."
        },
        {
            "role": "user", 
            "content": academic_prompt
        }
    ]
    
    data = { 
        "model": "gpt-4", 
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()
            response_json = response.json()
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return {
                    "success": True,
                    "response": response_json["choices"][0]["message"]["content"],
                    "error": None,
                    "attempts": attempt + 1
                }
            else:
                return {
                    "success": False,
                    "response": None,
                    "error": "No response from GPT",
                    "attempts": attempt + 1
                }
                
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "response": None,
                    "error": f"Request failed after {max_retries} attempts: {str(e)}",
                    "attempts": max_retries
                }
            time.sleep(2 ** attempt)
            
        except Exception as e:
            return {
                "success": False,
                "response": None,
                "error": f"Unexpected error: {str(e)}",
                "attempts": attempt + 1
            }

def read_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """读取YOLO格式的标签文件"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))
    return labels

def yolo_to_bbox(x_center: float, y_center: float, width: float, height: float, 
                img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """将YOLO格式转换为边界框坐标"""
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x1 = int(x_center_abs - width_abs / 2)
    y1 = int(y_center_abs - height_abs / 2)
    x2 = int(x_center_abs + width_abs / 2)
    y2 = int(y_center_abs + height_abs / 2)
    
    return x1, y1, x2, y2

def crop_image_with_bbox(image_path: str, bbox: Tuple[int, int, int, int], 
                        output_path: str) -> str:
    """根据边界框裁剪图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    x1, y1, x2, y2 = bbox
    # 确保边界框在图像范围内
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w))
    y2 = max(y1+1, min(y2, h))
    
    cropped = image[y1:y2, x1:x2]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped)
    return output_path

def parse_ground_truth(gt_answer: str) -> List[str]:
    """解析ground truth中的特征列表"""
    features = []
    lines = gt_answer.split('.')
    for line in lines:
        line = line.strip()
        if line:
            # 提取特征类型
            for feature_name, _ in segment_names_to_labels:
                if feature_name.lower() in line.lower():
                    features.append(feature_name)
                    break
    return list(set(features))  # 去重

def parse_gpt_response(gpt_response: str) -> List[str]:
    """解析GPT响应中的特征列表"""
    features = []
    response_lower = gpt_response.lower()
    for feature_name, _ in segment_names_to_labels:
        if feature_name.lower() in response_lower:
            features.append(feature_name)
    return list(set(features))

def compare_features(gpt_features: List[str], gt_features: List[str]) -> Tuple[List[str], List[str]]:
    """比较GPT检测的特征和ground truth，返回检测到的和未检测到的特征"""
    detected_features = []  # 特征A：能直接检测出来的
    missed_features = []    # 特征B：没有检测出来的
    
    for feature in gt_features:
        if feature in gpt_features:
            detected_features.append(feature)
        else:
            missed_features.append(feature)
    
    return detected_features, missed_features

def create_comparative_image(contextual_crop_path: str, comparative_crop_path: str, 
                           output_path: str) -> str:
    """创建对比图像"""
    img1 = cv2.imread(contextual_crop_path)
    img2 = cv2.imread(comparative_crop_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Cannot read one or both images for comparison")
    
    # 调整图像大小使其高度相同
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2:
        if h1 > h2:
            img2 = cv2.resize(img2, (int(w2 * h1 / h2), h1))
        else:
            img1 = cv2.resize(img1, (int(w1 * h2 / h1), h2))
    
    # 水平拼接
    combined = np.hstack([img1, img2])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, combined)
    return output_path

def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    """加载checkpoint文件"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_checkpoint(results: List[Dict], checkpoint_path: str) -> None:
    """保存checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_single_image(item: Dict, data_path: str, labels_path: str, 
                        output_dir: str, image_idx: int) -> Dict:
    """处理单张图片的完整流程 - 使用学术研究框架"""
    
    # 获取基本信息
    image_name = os.path.basename(item["image_name"])
    image_path = os.path.join(data_path, "images", image_name)
    label_path = os.path.join(labels_path, image_name.replace('.jpg', '.txt'))
    
    result = {
        "image_name": image_name,
        "image_path": image_path,
        "question": item["Question"],
        "ground_truth": item["Answer"],
        "steps": {},
        "final_result": "failed"
    }
    
    if not os.path.exists(image_path):
        result["error"] = f"Image not found: {image_path}"
        return result
    
    # 读取YOLO标签
    yolo_labels = read_yolo_labels(label_path)
    
    # 步骤1：第一次调用GPT-4，分析整张图片
    print(f"步骤1: 分析整张图片 {image_name}")
    step1_result = get_gpt_vision_response(image_path, "full_image_analysis")
    result["steps"]["step1"] = step1_result
    
    if not step1_result["success"]:
        result["error"] = "Step 1 failed: " + step1_result["error"]
        return result
    
    # 步骤2：比较GPT回答和ground truth
    print(f"步骤2: 比较检测结果")
    gt_features = parse_ground_truth(item["Answer"])
    gpt_features = parse_gpt_response(step1_result["gpt_answer"])
    
    detected_features, missed_features = compare_features(gpt_features, gt_features)
    
    comparison_context = f"""
CV Analysis Output: {step1_result["gpt_answer"]}
Reference Annotations: {item["Answer"]}

Available annotation categories: {[name for name, _ in segment_names_to_labels]}
"""
    
    step2_result = get_llm_text_response("comparison_analysis", comparison_context)
    result["steps"]["step2"] = {
        "comparison_result": step2_result,
        "detected_features": detected_features,
        "missed_features": missed_features,
        "gt_features": gt_features,
        "gpt_features": gpt_features
    }
    
    # 如果没有遗漏的特征，流程结束
    if not missed_features:
        result["final_result"] = "all_detected"
        return result
    
    # 为每个遗漏的特征进行后续步骤
    for feature_idx, missed_feature in enumerate(missed_features):
        print(f"处理遗漏特征: {missed_feature}")
        
        # 找到对应的边界框
        feature_label = None
        for name, label in segment_names_to_labels:
            if name == missed_feature:
                feature_label = label
                break
        
        if feature_label is None:
            continue
        
        # 从YOLO标签中找到对应的边界框
        feature_bboxes = []
        for class_id, x_center, y_center, width, height in yolo_labels:
            if class_id == feature_label:
                bbox = yolo_to_bbox(x_center, y_center, width, height, 
                                  item["image_width"], item["image_height"])
                feature_bboxes.append(bbox)
        
        if not feature_bboxes:
            continue
        
        # 使用第一个边界框（如果有多个的话）
        bbox = feature_bboxes[0]
        
        # 步骤3：使用contextual box裁剪图像
        print(f"步骤3: 使用contextual box分析 {missed_feature}")
        contextual_crop_path = os.path.join(output_dir, "crops", f"{image_idx}_{feature_idx}_contextual.jpg")
        crop_image_with_bbox(image_path, bbox, contextual_crop_path)
        
        step3_result = get_gpt_vision_response(contextual_crop_path, "cropped_region_analysis")
        
        if not result["steps"].get("step3"):
            result["steps"]["step3"] = {}
        result["steps"]["step3"][missed_feature] = step3_result
        
        if not step3_result["success"]:
            continue
        
        # 步骤4：验证检测结果
        print(f"步骤4: 验证检测结果")
        verification_context = f"Expected radiographic feature: {missed_feature}\nCV Analysis: {step3_result['gpt_answer']}"
        
        step4_result = get_llm_text_response("feature_verification", verification_context)
        
        if not result["steps"].get("step4"):
            result["steps"]["step4"] = {}
        result["steps"]["step4"][missed_feature] = step4_result
        
        if step4_result["success"] and "AGREE" in step4_result["response"].upper():
            # 正确识别，使用contextual box
            if not result["steps"].get("final_boxes"):
                result["steps"]["final_boxes"] = {}
            result["steps"]["final_boxes"][missed_feature] = {
                "type": "contextual",
                "bbox": bbox,
                "crop_path": contextual_crop_path
            }
            continue
        
        # 步骤5：使用comparative box
        print(f"步骤5: 使用comparative box分析 {missed_feature}")
        
        # 创建对比图像（扩大区域作为comparative box）
        margin = 20
        img_h, img_w = item["image_height"], item["image_width"]
        comparative_bbox = (
            max(0, bbox[0] - margin), 
            max(0, bbox[1] - margin), 
            min(img_w, bbox[2] + margin), 
            min(img_h, bbox[3] + margin)
        )
        comparative_crop_path = os.path.join(output_dir, "crops", f"{image_idx}_{feature_idx}_comparative.jpg")
        crop_image_with_bbox(image_path, comparative_bbox, comparative_crop_path)
        
        # 创建对比图像
        comparison_image_path = os.path.join(output_dir, "crops", f"{image_idx}_{feature_idx}_comparison.jpg")
        create_comparative_image(contextual_crop_path, comparative_crop_path, comparison_image_path)
        
        step5_result = get_gpt_vision_response(comparison_image_path, "comparative_analysis")
        
        if not result["steps"].get("step5"):
            result["steps"]["step5"] = {}
        result["steps"]["step5"][missed_feature] = step5_result
        
        if not step5_result["success"]:
            continue
        
        # 再次验证
        step5_verify_context = f"Expected radiographic feature: {missed_feature}\nCV Analysis: {step5_result['gpt_answer']}"
        
        step5_verify_result = get_llm_text_response("feature_verification", step5_verify_context)
        
        if step5_verify_result["success"] and "AGREE" in step5_verify_result["response"].upper():
            # 正确识别，使用comparative box
            if not result["steps"].get("final_boxes"):
                result["steps"]["final_boxes"] = {}
            result["steps"]["final_boxes"][missed_feature] = {
                "type": "comparative",
                "bbox": comparative_bbox,
                "crop_path": comparison_image_path
            }
            continue
        
        # 步骤6：基于ground truth进行解释
        print(f"步骤6: 基于ground truth解释 {missed_feature}")
        step6_result = get_llm_text_response("educational_description", missed_feature)
        
        if not result["steps"].get("step6"):
            result["steps"]["step6"] = {}
        result["steps"]["step6"][missed_feature] = step6_result
        
        # 最终使用ground truth解释
        if not result["steps"].get("final_boxes"):
            result["steps"]["final_boxes"] = {}
        result["steps"]["final_boxes"][missed_feature] = {
            "type": "ground_truth_explanation",
            "bbox": bbox,
            "explanation": step6_result["response"] if step6_result["success"] else "Failed to generate explanation"
        }
    
    result["final_result"] = "completed"
    return result

def process_dental_dataset(json_path: str, data_path: str, labels_path: str, 
                         output_path: str = None, start_idx: int = 0, 
                         end_idx: int = None) -> None:
    """处理牙科数据集"""
    
    # 读取JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # 确定处理范围
    if end_idx is None:
        end_idx = len(data_list)
    
    data_list = data_list[start_idx:end_idx]
    
    # 设置输出路径
    if output_path is None:
        output_path = '/hpc2hdd/home/yfan546/workplace/xray_teeth/r1/mmoral_reasoning_data/step_1_output'
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "crops"), exist_ok=True)
    
    # checkpoint文件路径
    checkpoint_file = os.path.join(output_path, "multi_step_results.json")
    
    # 加载已有的结果
    existing_results = load_checkpoint(checkpoint_file)
    processed_indices = {result.get("index", -1) for result in existing_results}
    
    print(f"开始处理 {len(data_list)} 个样本...")
    print(f"已处理样本数: {len(existing_results)}")
    
    results = existing_results.copy()
    
    for idx, item in enumerate(tqdm(data_list, desc="Processing images")):
        current_idx = start_idx + idx
        
        # 跳过已处理的样本
        if current_idx in processed_indices:
            print(f"跳过已处理的样本 {current_idx}")
            continue
            
        try:
            print(f"\n处理第 {idx+1}/{len(data_list)} 个样本: {item['image_name']}")
            
            result = process_single_image(item, data_path, labels_path, output_path, current_idx)
            result["index"] = current_idx
            results.append(result)
            
            # 每处理10个样本保存一次checkpoint
            if len(results) % 2 == 0:
                print(f"保存checkpoint，当前已处理 {len(results)} 个样本")
                save_checkpoint(results, checkpoint_file)
                
        except Exception as e:
            print(f"处理第 {idx} 个样本时出错: {str(e)}")
            error_result = {
                "index": current_idx,
                "image_name": item.get("image_name", "unknown"),
                "error": str(e),
                "final_result": "error"
            }
            results.append(error_result)
            continue
    
    # 保存最终结果
    save_checkpoint(results, checkpoint_file)
    
    # 打印统计信息
    print_final_statistics(results)

def print_final_statistics(results: List[Dict]) -> None:
    """打印最终统计信息"""
    total = len(results)
    completed = sum(1 for r in results if r.get("final_result") == "completed")
    all_detected = sum(1 for r in results if r.get("final_result") == "all_detected")
    failed = sum(1 for r in results if r.get("final_result") == "failed")
    errors = sum(1 for r in results if r.get("final_result") == "error")
    
    print("\n" + "="*60)
    print("多步骤分析结果统计")
    print("="*60)
    print(f"总处理样本数: {total}")
    print(f"完成多步骤分析: {completed} ({completed/total*100:.1f}%)")
    print(f"第一步就全部检测正确: {all_detected} ({all_detected/total*100:.1f}%)")
    print(f"处理失败: {failed} ({failed/total*100:.1f}%)")
    print(f"出现错误: {errors} ({errors/total*100:.1f}%)")
    
    # 统计使用的box类型
    contextual_count = 0
    comparative_count = 0
    explanation_count = 0
    
    for result in results:
        if "steps" in result and "final_boxes" in result["steps"]:
            for feature, box_info in result["steps"]["final_boxes"].items():
                if box_info["type"] == "contextual":
                    contextual_count += 1
                elif box_info["type"] == "comparative":
                    comparative_count += 1
                elif box_info["type"] == "ground_truth_explanation":
                    explanation_count += 1
    
    print(f"\n最终采用的方法统计:")
    print(f"Contextual box: {contextual_count}")
    print(f"Comparative box: {comparative_count}")
    print(f"Ground truth explanation: {explanation_count}")
    print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多步骤GPT牙科X光图片特征检测分析 - 学术研究版本')
    parser.add_argument('--json_path', type=str, 
                        default="/hpc2hdd/home/yfan546/workplace/xray_teeth/r1/source_data/Multimodal_data_Dental_Conditions_Detection_2025_Romania.json",
                        help='JSON文件路径')
    parser.add_argument('--data_path', type=str, 
                        default="/hpc2hdd/home/yfan546/workplace/xray_teeth/r1/source_data/4_1.6k/all",
                        help='数据根目录路径')
    parser.add_argument('--labels_path', type=str, 
                        default="/hpc2hdd/home/yfan546/workplace/xray_teeth/r1/source_data/4_1.6k/all/labels",
                        help='标签文件目录路径')
    parser.add_argument('--output_path', type=str, 
                        default='/hpc2hdd/home/yfan546/workplace/xray_teeth/r1/mmoral_reasoning_data/step_1_output',
                        help='输出目录路径')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='开始处理的索引')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='结束处理的索引')
    
    args = parser.parse_args()
    
    # 执行处理
    process_dental_dataset(
        json_path=args.json_path,
        data_path=args.data_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )

if __name__ == "__main__":
    main()
