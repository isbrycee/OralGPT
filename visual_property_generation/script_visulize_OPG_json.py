# 生成所有框
# python d_vis_oral_json.py --json_dir ../unlabeled_data/MM-Oral-OPG-jsons/ --img_dir ../unlabeled_data/MM-Oral-OPG-images/  --mode random --num_samples 5

# 只生成”牙号和象限“框
#python d_vis_oral_json.py --json_dir ../unlabeled_data/MM-Oral-OPG-jsons/ --img_dir ../unlabeled_data/MM-Oral-OPG-images/  --mode random --num_samples 5  --vis_mode teeth_quadrants

# 只生成conditions和jawbones框
#python d_vis_oral_json.py --json_dir ../unlabeled_data/MM-Oral-OPG-jsons/ --img_dir ../unlabeled_data/MM-Oral-OPG-images/  --mode random --num_samples 5  --vis_mode conditions_jawbones

# 生成以上三种模式的框
#python d_vis_oral_json.py --json_dir ../unlabeled_data/MM-Oral-OPG-jsons_output/ --img_dir ../unlabeled_data/MM-Oral-OPG-images/  --mode random --num_samples 5  --vis_mode multi

import os
import json
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
from matplotlib.path import Path
from pycocotools import mask as mask_utils
import cv2

def load_json_data(json_file):
    """加载单个JSON文件并返回数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def validate_bbox(bbox, format='xyxy'):
    """验证边界框是否有效"""
    if bbox is None or len(bbox) == 0:
        return False
    
    if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
        return True
    
    if format in ['xyxy', 'xywh', 'cxcywh'] and len(bbox) < 4:
        return False
    
    try:
        for val in bbox:
            float(val)
        return True
    except (ValueError, TypeError):
        return False

def draw_bbox(ax, bbox, format='xyxy', **kwargs):
    """绘制边界框"""
    if isinstance(bbox, list) and len(bbox) > 0 and isinstance(bbox[0], list):
        rects = []
        for single_bbox in bbox:
            rect = draw_bbox(ax, single_bbox, format, **kwargs)
            if rect:
                rects.append(rect)
        return rects
    
    if not validate_bbox(bbox, format):
        return None
    
    try:
        if format == 'xyxy':
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, **kwargs)
        elif format == 'xywh':
            x, y, width, height = bbox
            rect = patches.Rectangle((x, y), width, height, **kwargs)
        elif format == 'cxcywh':
            cx, cy, width, height = bbox
            x = cx - width / 2
            y = cy - height / 2
            rect = patches.Rectangle((x, y), width, height, **kwargs)
        else:
            raise ValueError(f"Unsupported bbox format: {format}")
        
        ax.add_patch(rect)
        return rect
    except (ValueError, TypeError) as e:
        print(f"绘制边界框出错: {e}, bbox: {bbox}")
        return None

def decode_rle(rle_data, img_height, img_width):
    """解码RLE格式的分割数据"""
    try:
        if not rle_data or 'size' not in rle_data or 'counts' not in rle_data:
            return None
        
        binary_mask = mask_utils.decode(rle_data)
        return binary_mask
    except Exception as e:
        print(f"解码RLE数据出错: {e}")
        return None

def save_image_with_annotations(data, img_dir, output_path, bbox_format, vis_mode="all", img_id=None, file_name=None):
    """可视化单个图像及其标注，并保存结果"""
    if img_id is None:
        img_id = data.get("image_id", "unknown")
    if file_name is None:
        file_name = data.get("file_name", "unknown.jpg")
    
    img_path = os.path.join(img_dir, file_name)
    
    # 获取图像尺寸
    img_width = data.get("image_width", 1000)
    img_height = data.get("image_height", 1000)
    
    # 创建图形并关闭自动调整
    plt.rcParams.update({'figure.autolayout': False})
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 使用OpenCV读取图像并保持原始对比度
    try:
        # 使用IMREAD_UNCHANGED保持原始位深度
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception("图像读取失败")
        
        # 如果是灰度图，保持灰度；如果是彩色图，转换为RGB
        if len(img.shape) == 2:
            # 灰度图 - 创建更好的色彩映射以保持对比度
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            # 彩色图 - BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        
        # 设置图像边界，确保完全显示
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)  # 反转Y轴以匹配图像坐标系
    except Exception as e:
        print(f"加载图像 {img_path} 失败: {e}")
        # 如果图像无法加载，创建一个空白画布，使用图像尺寸
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # 反转Y轴以匹配图像坐标系
    
    # 关闭轴显示
    ax.axis('off')
    
    # 根据可视化模式设置标题
    mode_descriptions = {
        "all": "全部内容",
        "teeth_quadrants": "牙号和象限",
        "conditions_jawbones": "疾病和颌骨"
    }
    
    title = f"图像ID: {img_id}, 文件名: {file_name}\n"
    title += f"格式: {bbox_format}, 模式: {mode_descriptions.get(vis_mode, vis_mode)}"
    ax.set_title(title)
    
    # 获取图像的标注 - 牙齿
    teeth = data.get("properties", {}).get("Teeth", [])
    
    # 决定显示内容
    show_teeth_boxes = vis_mode in ["all", "teeth_quadrants"]
    show_conditions = vis_mode in ["all", "conditions_jawbones"]
    show_quadrants = vis_mode in ["all", "teeth_quadrants"]
    
    # 绘制牙齿标注
    if teeth:
        for i, tooth in enumerate(teeth):
            color = plt.cm.tab10(i % 10)
            
            # 绘制牙齿边界框和ID
            if show_teeth_boxes and 'bbox' in tooth:
                bbox = tooth['bbox']
                if validate_bbox(bbox, bbox_format):
                    draw_bbox(ax, bbox, format=bbox_format, fill=False, 
                             edgecolor=color, linewidth=1.5, alpha=0.8)
                    
                    # 显示牙齿ID
                    if 'tooth_id' in tooth:
                        try:
                            if bbox_format == 'xywh':
                                text_x, text_y = bbox[0], bbox[1] - 5
                            elif bbox_format == 'cxcywh':
                                cx, cy, w, h = bbox
                                text_x, text_y = cx - w/2, cy - h/2 - 5
                            else:  # xyxy
                                text_x, text_y = bbox[0], bbox[1] - 5
                            
                            ax.text(
                                text_x, text_y, 
                                f"ID:{tooth['tooth_id']} ({tooth.get('score', 0):.2f})",
                                bbox=dict(facecolor=color, alpha=0.5),
                                fontsize=8
                            )
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"绘制牙齿标注出错: {e}, bbox: {bbox}")
            
            # 绘制疾病条件
            if show_conditions and "conditions" in tooth:
                conditions = tooth.get("conditions", {})
                for cond_name, cond_data in conditions.items():
                    # 检查是否是布尔值 - 处理类似 "Recommended extraction": true 的情况
                    if isinstance(cond_data, bool):
                        if cond_data and show_teeth_boxes:  # 只有当显示牙齿框时才显示布尔条件
                            # 找到牙齿框的位置
                            if 'bbox' in tooth and validate_bbox(tooth['bbox'], bbox_format):
                                if bbox_format == 'xyxy':
                                    text_x, text_y = tooth['bbox'][0], tooth['bbox'][3] + 15
                                elif bbox_format == 'xywh':
                                    text_x, text_y = tooth['bbox'][0], tooth['bbox'][1] + tooth['bbox'][3] + 15
                                else:  # cxcywh
                                    cx, cy, w, h = tooth['bbox']
                                    text_x, text_y = cx - w/2, cy + h/2 + 15
                                
                                ax.text(
                                    text_x, text_y, 
                                    f"{cond_name}",
                                    bbox=dict(facecolor='red', alpha=0.3),
                                    fontsize=8, color='darkred'
                                )
                        continue
                    
                    if cond_data.get("present", False) and "bbox" in cond_data:
                        cond_bbox = cond_data["bbox"]
                        
                        # 处理bbox可能是列表的列表的情况
                        if isinstance(cond_bbox, list) and len(cond_bbox) > 0 and isinstance(cond_bbox[0], list):
                            for idx, single_bbox in enumerate(cond_bbox):
                                if validate_bbox(single_bbox, bbox_format):
                                    draw_bbox(ax, single_bbox, format=bbox_format, fill=False, 
                                             edgecolor='red', linewidth=1, linestyle='--', alpha=0.7)
                                    
                                    try:
                                        text_x, text_y = single_bbox[0], single_bbox[1] - 5
                                        
                                        # 获取分数
                                        score = 0
                                        if 'score' in cond_data:
                                            score_value = cond_data['score']
                                            if isinstance(score_value, list) and len(score_value) > idx:
                                                score = score_value[idx]
                                            else:
                                                score = score_value
                                        
                                        # 显示标签
                                        label = f"{cond_name}"
                                        if not show_teeth_boxes:
                                            label = f"{tooth.get('tooth_id', 'Unknown')}: {label}"
                                            
                                        ax.text(
                                            text_x, text_y, 
                                            f"{label} ({score:.2f})",
                                            bbox=dict(facecolor='red', alpha=0.3),
                                            fontsize=8, color='darkred'
                                        )
                                    except (ValueError, TypeError, IndexError) as e:
                                        print(f"绘制疾病标注出错: {e}, bbox: {single_bbox}")
                        else:
                            if validate_bbox(cond_bbox, bbox_format):
                                draw_bbox(ax, cond_bbox, format=bbox_format, fill=False, 
                                         edgecolor='red', linewidth=1, linestyle='--', alpha=0.7)
                                
                                try:
                                    text_x, text_y = cond_bbox[0], cond_bbox[1] - 5
                                    
                                    # 显示标签
                                    label = f"{cond_name}"
                                    if not show_teeth_boxes:
                                        label = f"{tooth.get('tooth_id', 'Unknown')}: {label}"
                                        
                                    ax.text(
                                        text_x, text_y, 
                                        f"{label} ({cond_data.get('score', 0):.2f})",
                                        bbox=dict(facecolor='red', alpha=0.3),
                                        fontsize=8, color='darkred'
                                    )
                                except (ValueError, TypeError, IndexError) as e:
                                    print(f"绘制疾病标注出错: {e}, bbox: {cond_bbox}")
                        
                        # 如果有分割数据，绘制分割
                        if "segmentation" in cond_data:
                            segmentation = cond_data["segmentation"]
                            
                            # 处理RLE编码的分割
                            if isinstance(segmentation, dict) and 'size' in segmentation and 'counts' in segmentation:
                                try:
                                    binary_mask = decode_rle(segmentation, img_height, img_width)
                                    if binary_mask is not None:
                                        h, w = binary_mask.shape
                                        # 使用很低的alpha值以避免覆盖原图细节
                                        ax.imshow(binary_mask, alpha=0.15, cmap='Reds', extent=[0, w, h, 0])
                                except Exception as e:
                                    print(f"绘制RLE分割出错: {e}")
                            elif isinstance(segmentation, list) and all(isinstance(item, dict) for item in segmentation):
                                # 多个RLE编码
                                for seg in segmentation:
                                    try:
                                        binary_mask = decode_rle(seg, img_height, img_width)
                                        if binary_mask is not None:
                                            h, w = binary_mask.shape
                                            ax.imshow(binary_mask, alpha=0.15, cmap='Reds', extent=[0, w, h, 0])
                                    except Exception as e:
                                        print(f"绘制多个RLE分割出错: {e}")
                            elif isinstance(segmentation, list) and len(segmentation) >= 6:
                                # 传统多边形格式
                                try:
                                    poly = np.array(segmentation).reshape((-1, 2))
                                    ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color='red')
                                    ax.plot(poly[:, 0], poly[:, 1], color='red', linewidth=0.8, alpha=0.5)
                                except Exception as e:
                                    print(f"绘制多边形分割出错: {e}")
    
    # 获取并绘制象限标注
    if show_quadrants:
        quadrants_data = data.get("properties", {}).get("Quadrants", [])
        if quadrants_data:
            for quadrant in quadrants_data:
                # 如果象限存在且有边界框
                if quadrant.get("present", False) and "bbox" in quadrant:
                    q_bbox = quadrant["bbox"]
                    if validate_bbox(q_bbox, bbox_format):
                        draw_bbox(ax, q_bbox, format=bbox_format, fill=False, 
                                 edgecolor='green', linewidth=0.8, linestyle=':', alpha=0.6)
                        
                        try:
                            text_x, text_y = q_bbox[0], q_bbox[1]
                            
                            quadrant_name = quadrant.get("quadrant", "Unknown")
                            ax.text(
                                text_x, text_y, 
                                f"{quadrant_name} ({quadrant.get('score', 0):.2f})",
                                color='green', fontsize=8, alpha=0.7
                            )
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"绘制象限标注出错: {e}, bbox: {q_bbox}")
                
                # 如果有分割数据，绘制分割
                if "segmentation" in quadrant:
                    segmentation = quadrant["segmentation"]
                    if isinstance(segmentation, dict) and 'size' in segmentation and 'counts' in segmentation:
                        try:
                            binary_mask = decode_rle(segmentation, img_height, img_width)
                            if binary_mask is not None:
                                h, w = binary_mask.shape
                                # 使用更低的alpha值
                                ax.imshow(binary_mask, alpha=0.08, cmap='Greens', extent=[0, w, h, 0])
                        except Exception as e:
                            print(f"绘制象限RLE分割出错: {e}")
    
    # 获取并绘制颌骨标注
    if show_conditions:
        jawbones_data = data.get("properties", {}).get("JawBones", [])
        if jawbones_data:
            for jawbone_obj in jawbones_data:
                if "conditions" in jawbone_obj:
                    conditions = jawbone_obj["conditions"]
                    for bone_cond_name, bone_cond_data in conditions.items():
                        if "bbox" in bone_cond_data:
                            bone_bbox = bone_cond_data["bbox"]
                            
                            # 处理bbox可能是列表的列表的情况
                            if isinstance(bone_bbox, list) and len(bone_bbox) > 0 and isinstance(bone_bbox[0], list):
                                for idx, single_bbox in enumerate(bone_bbox):
                                    if validate_bbox(single_bbox, bbox_format):
                                        draw_bbox(ax, single_bbox, format=bbox_format, fill=False, 
                                                 edgecolor='blue', linewidth=1, linestyle='-', alpha=0.6)
                                        
                                        try:
                                            text_x, text_y = single_bbox[0], single_bbox[1] - 5
                                            
                                            # 获取分数
                                            score = 0
                                            if 'score' in bone_cond_data:
                                                score_value = bone_cond_data['score']
                                                if isinstance(score_value, list) and len(score_value) > idx:
                                                    score = score_value[idx]
                                                else:
                                                    score = score_value
                                            
                                            ax.text(
                                                text_x, text_y, 
                                                f"{bone_cond_name} ({score:.2f})",
                                                bbox=dict(facecolor='blue', alpha=0.2),
                                                fontsize=8, color='darkblue'
                                            )
                                        except (ValueError, TypeError, IndexError) as e:
                                            print(f"绘制颌骨标注出错: {e}, bbox: {single_bbox}")
                            else:
                                if validate_bbox(bone_bbox, bbox_format):
                                    draw_bbox(ax, bone_bbox, format=bbox_format, fill=False, 
                                             edgecolor='blue', linewidth=1, linestyle='-', alpha=0.6)
                                    
                                    try:
                                        text_x, text_y = bone_bbox[0], bone_bbox[1] - 5
                                        
                                        ax.text(
                                            text_x, text_y, 
                                            f"{bone_cond_name} ({bone_cond_data.get('score', 0):.2f})",
                                            bbox=dict(facecolor='blue', alpha=0.2),
                                            fontsize=8, color='darkblue'
                                        )
                                    except (ValueError, TypeError, IndexError) as e:
                                        print(f"绘制颌骨标注出错: {e}, bbox: {bone_bbox}")
                        
                        # 如果有分割数据，绘制分割
                        if "segmentation" in bone_cond_data:
                            segmentation = bone_cond_data["segmentation"]
                            
                            # 处理可能的多个RLE编码
                            if isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], dict):
                                for seg in segmentation:
                                    try:
                                        binary_mask = decode_rle(seg, img_height, img_width)
                                        if binary_mask is not None:
                                            h, w = binary_mask.shape
                                            # 使用更低的alpha值
                                            ax.imshow(binary_mask, alpha=0.08, cmap='Blues', extent=[0, w, h, 0])
                                    except Exception as e:
                                        print(f"绘制多个颌骨RLE分割出错: {e}")
                            # 处理单个RLE编码
                            elif isinstance(segmentation, dict) and 'size' in segmentation and 'counts' in segmentation:
                                try:
                                    binary_mask = decode_rle(segmentation, img_height, img_width)
                                    if binary_mask is not None:
                                        h, w = binary_mask.shape
                                        ax.imshow(binary_mask, alpha=0.08, cmap='Blues', extent=[0, w, h, 0])
                                except Exception as e:
                                    print(f"绘制单个颌骨RLE分割出错: {e}")
    
    # 获取并绘制Missing teeth标注
    if show_conditions:
        missing_teeth = data.get("properties", {}).get("Missing teeth", [])
        if missing_teeth:
            for missing_tooth in missing_teeth:
                if "bbox" in missing_tooth:
                    missing_bbox = missing_tooth["bbox"]
                    if validate_bbox(missing_bbox, bbox_format):
                        draw_bbox(ax, missing_bbox, format=bbox_format, fill=False, 
                                 edgecolor='purple', linewidth=1, linestyle='-', alpha=0.6)
                        
                        try:
                            text_x, text_y = missing_bbox[0], missing_bbox[1] - 5
                            
                            side = missing_tooth.get("side", "Unknown")
                            ax.text(
                                text_x, text_y, 
                                f"Missing tooth ({missing_tooth.get('score', 0):.2f})",
                                bbox=dict(facecolor='purple', alpha=0.2),
                                fontsize=8, color='darkviolet'
                            )
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"绘制缺失牙齿标注出错: {e}, bbox: {missing_bbox}")
                
                # 绘制分割
                if "segmentation" in missing_tooth:
                    segmentation = missing_tooth["segmentation"]
                    if isinstance(segmentation, dict) and 'size' in segmentation and 'counts' in segmentation:
                        try:
                            binary_mask = decode_rle(segmentation, img_height, img_width)
                            if binary_mask is not None:
                                h, w = binary_mask.shape
                                ax.imshow(binary_mask, alpha=0.08, cmap='Purples', extent=[0, w, h, 0])
                        except Exception as e:
                            print(f"绘制缺失牙齿RLE分割出错: {e}")
    
    # 保存图像，使用高DPI和无损格式
    try:
        # 移除多余的空白边距，但保留一些边距用于标题
        plt.tight_layout(pad=1)
        
        # 使用更高的DPI和PNG格式以保持更好的质量
        plt.savefig(output_path, dpi=300, format='png', 
                  bbox_inches='tight', pad_inches=0.1, 
                  transparent=False, facecolor='black', edgecolor='none')
        plt.close(fig)
        
        # 对保存后的图像可选地进一步处理，以确保对比度
        try:
            # 读取刚保存的图像
            saved_img = cv2.imread(output_path)
            if saved_img is not None:
                # 应用一些对比度增强，但不要太激进
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                
                # 如果是彩色图像，分别增强每个通道
                if len(saved_img.shape) == 3:
                    lab = cv2.cvtColor(saved_img, cv2.COLOR_BGR2Lab)
                    l, a, b = cv2.split(lab)
                    cl = clahe.apply(l)
                    merged = cv2.merge((cl, a, b))
                    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_Lab2BGR)
                else:
                    # 灰度图像
                    enhanced_img = clahe.apply(saved_img)
                
                # 保存增强后的图像
                cv2.imwrite(output_path, enhanced_img)
        except Exception as e:
            print(f"增强图像后处理失败: {e}, 使用原始保存的图像")
        
        return True
    except Exception as e:
        print(f"保存可视化图像失败: {e}")
        plt.close(fig)
        return False

def create_visualizations_for_file(data, img_dir, output_dir, bbox_format, file_index):
    """为单个文件创建三种不同模式的可视化"""
    file_name = data.get("file_name", f"unknown_{file_index}.jpg")
    base_name = os.path.basename(file_name).replace('.jpg', '').replace('.png', '')
    
    results = []
    
    try:
        # 1. 显示所有内容
        output_path = os.path.join(output_dir, f"vis_{file_index}_{base_name}_all.png")
        result1 = save_image_with_annotations(
            data, img_dir, output_path, bbox_format, vis_mode="all", file_name=file_name
        )
        results.append(result1)
        
        # 2. 只显示牙号和象限
        output_path = os.path.join(output_dir, f"vis_{file_index}_{base_name}_teeth_quadrants.png")
        result2 = save_image_with_annotations(
            data, img_dir, output_path, bbox_format, vis_mode="teeth_quadrants", file_name=file_name
        )
        results.append(result2)
        
        # 3. 只显示牙齿疾病和颌骨
        output_path = os.path.join(output_dir, f"vis_{file_index}_{base_name}_conditions_jawbones.png")
        result3 = save_image_with_annotations(
            data, img_dir, output_path, bbox_format, vis_mode="conditions_jawbones", file_name=file_name
        )
        results.append(result3)
        
        return any(results)
    except Exception as e:
        print(f"创建文件 {file_name} 的可视化失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='可视化牙齿JSON数据')
    parser.add_argument('--json_dir', type=str, required=True, help='包含JSON文件的目录')
    parser.add_argument('--img_dir', type=str, required=True, help='包含图像文件的目录')
    parser.add_argument('--output_dir', type=str, default='visualization', help='输出可视化结果的目录')
    parser.add_argument('--mode', type=str, choices=['random', 'all'], default='random', help='可视化模式: random (随机选择) 或 all (所有图像)')
    parser.add_argument('--num_samples', type=int, default=5, help='随机模式下要可视化的图像数量')
    parser.add_argument('--bbox_format', type=str, choices=['xywh', 'cxcywh', 'xyxy'], default='xyxy', help='边界框格式: xywh, cxcywh 或 xyxy')
    parser.add_argument('--vis_mode', type=str, choices=['all', 'teeth_quadrants', 'conditions_jawbones', 'multi'], 
                       default='multi', help='可视化内容模式')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
    if not json_files:
        print(f"错误: 在 {args.json_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    if args.mode == 'random':
        # 随机选择指定数量的JSON文件
        sample_size = min(args.num_samples, len(json_files))
        selected_files = random.sample(json_files, sample_size)
        print(f"随机选择了 {sample_size} 个文件进行可视化")
    else:
        # 使用所有JSON文件
        selected_files = json_files
        print(f"将可视化所有 {len(selected_files)} 个文件")
    
    # 处理选中的文件
    successful_saves = 0
    for i, json_file in enumerate(selected_files):
        try:
            # 加载JSON数据
            data = load_json_data(json_file)
            
            if args.vis_mode == 'multi':
                # 生成三种不同的可视化模式
                if create_visualizations_for_file(data, args.img_dir, args.output_dir, args.bbox_format, i+1):
                    successful_saves += 1
                    print(f"成功为文件 {i+1}/{len(selected_files)} 创建多模式可视化")
            else:
                # 生成单一模式的可视化
                base_name = os.path.basename(json_file)
                file_name = data.get("file_name", "unknown.jpg")
                output_filename = f"vis_{args.bbox_format}_{args.vis_mode}_{i+1}_{base_name.replace('.json', '.png')}"
                output_path = os.path.join(args.output_dir, output_filename)
                
                if save_image_with_annotations(
                    data, args.img_dir, output_path, args.bbox_format, vis_mode=args.vis_mode, file_name=file_name
                ):
                    successful_saves += 1
                    print(f"成功保存可视化结果到: {output_path} (文件 {i+1}/{len(selected_files)})")
            
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    print(f"\n总共成功保存了 {successful_saves}/{len(selected_files)} 张可视化图像")
    print(f"可视化结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
