import os
import json
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms
import traceback
from tqdm import tqdm
import sys
sys.path.append("/home/jinghao/projects/dental_plague_detection/DINO-main")
import datasets.transforms as T


# 定义类别常量
TEETH_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Crown"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "Deep Caries"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "RCT"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "Restoration"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "Caries"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "Normal"},
]

DISEASE_CATEGORIES = [
    "Pulpitis",          # 牙髓炎 
    "Impacted tooth",    # 阻生齿
    "Apical periodontitis", # 根尖周炎
    "Bone loss",         # 骨质流失
    "Alternation between primary and permanent teeth", # 乳恒牙交替期 score threhold = 0.8
    "Caries",            # 龋齿
    "Periodontitis",     # 牙周炎
]

# 置信度阈值
SCORE_THRESHOLD = 0.55

SCORE_THRESHOLD_Alternation = 0.8
SCORE_THRESHOLD_Impacted = 0.95
SCORE_THRESHOLD_Pulpitis = 0.8
SCORE_THRESHOLD_Periodontitis = 0.8
SCORE_THRESHOLD_Apical_Periodontitis = 0.78
SCORE_THRESHOLD_Bone_loss = 0.75
SCORE_THRESHOLD_Caries = 0.7

def convert_numpy_types(obj):
    """
    递归转换所有NumPy类型为标准Python类型
    
    Args:
        obj: 任意Python对象
        
    Returns:
        转换后的对象，所有NumPy类型都被转换为原生Python类型
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def load_vit_model(model_type):
    """
    加载ViT模型
    
    Args:
        model_type: 模型类型 (6diseases, 7diseases)
    
    Returns:
        model: 加载好的ViT模型
        predictor: 预测器
        id2name: 类别ID到名称的映射
    """
    sys.path.append("/home/jinghao/projects/x-ray-VLM/ViT-pytorch")  # vit
    from models_ViT.modeling import VisionTransformer, CONFIGS
    
    # 根据模型类型选择配置
    if model_type == "7diseases":
        config = CONFIGS["ViT-L_16"]
        model_weights = "/home/jinghao/projects/x-ray-VLM/ViT-pytorch/output/Teeth_Visual_Experts_ViT_L_periapical_images_cls_7diseases.bin"
        num_classes = 100
    
    img_size = 512
    
    # 初始化模型
    model = VisionTransformer(config, img_size=img_size, zero_head=False, num_classes=num_classes)
    
    # 加载模型权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state_dict=torch.load(model_weights, map_location='cuda'), strict=False)
    model.to(device)
    model.eval()
    
    # 类别ID到名称的映射
    if model_type == "7diseases":
        id2name = {i: name for i, name in enumerate(DISEASE_CATEGORIES)}
    else:  # 6diseases
        id2name = {cat["id"]: cat["name"] for cat in TEETH_CATEGORIES}
    
    return model, None, id2name

def load_maskdino_model(model_type):
    """
    加载MaskDINO模型
    
    Args:
        model_type: 模型类型 (6diseases, 7diseases)
    
    Returns:
        model: 加载好的MaskDINO模型
        predictor: 预测器
        id2name: 类别ID到名称的映射
    """
    import sys
    sys.path.append("/home/jinghao/projects/dental_plague_detection/MaskDINO")
    
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config
    from demo.predictor import VisualizationDemo

    
    if model_type == "6diseases":
        config_file = "/home/jinghao/projects/dental_plague_detection/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_periapical_x-ray_6diseases.yaml"
        model_weights = "/home/jinghao/projects/dental_plague_detection/MaskDINO/Teeth_Visual_Experts_Maskdino_Swinl_periapical_images_6diseases.pth"
        category_map_path = "/home/jinghao/projects/x-ray-VLM/dataset/periapical_x-ray_ins_seg_num1899/output_coco.json"
    else:
        raise ValueError(f"Unknown MaskDINO model type: {model_type}")
    
    # 配置MaskDINO
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    
    # 创建预测器
    predictor = VisualizationDemo(cfg)
    
    # 加载类别名称映射
    id2name = {}
    if os.path.exists(category_map_path):
        with open(category_map_path) as f:
            data = json.load(f)
            if 'categories' in data:
                id2name = {cat['id']: cat['name'] for cat in data['categories']}
            else:
                # 尝试直接使用，但跳过非数字键
                for k, v in data.items():
                    try:
                        id2name[int(k)] = v
                    except ValueError:
                        continue
    
    return cfg, predictor, id2name

class DentalVisualExpert:
    def __init__(self, model_type):
        """
        初始化牙科视觉专家模型
        
        Args:
            model_type: 模型类型 ("6diseases" 或 "7diseases")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # 加载模型
        if model_type == "6diseases":
            # 使用MaskDINO模型加载6diseases模型
            try:
                cfg, predictor, id2name = load_maskdino_model(model_type)
                self.predictor = predictor
                self.id2name = id2name
                print(f"Loaded {model_type} MaskDINO model successfully")
                
                # 图像预处理转换
                self.transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            except Exception as e:
                print(f"Failed to load MaskDINO model: {e}, trying ViT model instead")
                model, _, id2name = load_vit_model(model_type)
                self.model = model
                self.id2name = id2name
                self.predictor = None
                print(f"Loaded {model_type} ViT model successfully")
        
        elif model_type == "7diseases":
            # 使用ViT模型加载7diseases模型
            try:
                model, _, id2name = load_vit_model(model_type)
                self.model = model
                self.id2name = id2name
                self.predictor = None
                print(f"Loaded {model_type} ViT model successfully")
                
                self.transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
                
                
            except Exception as e:
                raise ValueError(f"Unknown ViT model type: {e}")

    def preprocess_image(self, image_path):
        """预处理图像并返回原始图像和变换后的图像"""
        image = Image.open(image_path).convert("RGB")
        transformed_image, _ = self.transform(image, None)
        return image, transformed_image
    
    def preprocess_vit(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transformed_image = self.transform(image)
        return image, transformed_image

    def detect_6diseases(self, image_path):
        """
        检测图像中的6种牙齿状况
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            检测结果字典
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return {}
        
        # 获取图像尺寸
        height, width = img.shape[:2]
        
        # 初始化结果字典
        result = {
            "Location": {}
        }
        
        if self.predictor:
            # 使用MaskDINO预测器
            predictions, _ = self.predictor.run_on_image(img)
            
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                
                # 获取边界框、分数和类别
                if len(instances) > 0:
                    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
                    scores = instances.scores.numpy() if instances.has("scores") else []
                    class_ids = instances.pred_classes.numpy() if instances.has("pred_classes") else []
                    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
                    
                    # 只处理置信度高的检测结果
                    high_conf_indices = np.where(scores > SCORE_THRESHOLD)[0]
                    
                    for idx in high_conf_indices:
                        box = boxes[idx]
                        score = float(scores[idx])
                        class_id = int(class_ids[idx])
                        mask = masks[idx] if len(masks) > idx else None
                        
                        # 获取类别名称
                        category_name = self.id2name.get(class_id + 1)  # +1 因为COCO格式从1开始
                        if not category_name:
                            continue
                        if category_name == "Normal": # filter normal teeth
                            continue
                        # 将xyxy格式的边界框转换为cx,cy,w,h格式
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        cx = x1 + w / 2
                        cy = y1 + h / 2
                        
                        bbox = [float(cx), float(cy), float(w), float(h)]
                        
                        # 将掩码转换为轮廓
                        segmentation = []
                        if mask is not None:
                            segmentation = self._encode_mask(mask)
                        
                        # 添加到结果字典
                        if category_name not in result["Location"]:
                            result["Location"][category_name] = {
                                "bbox": [],
                                "segmentation": [],
                                "score": []
                            }
                        
                        result["Location"][category_name]["bbox"].append(bbox)
                        result["Location"][category_name]["segmentation"].append(segmentation)
                        result["Location"][category_name]["score"].append(round(score, 2))
        
        else:
            pass
        
        return result

    def detect_7diseases(self, image_path, seg_results=None):
        """
        检测图像中的7种牙科疾病
        
        Args:
            image_path: 图像文件路径
            seg_results: 分割模型的结果，用于条件过滤
            
        Returns:
            检测结果字典
        """
        # 初始化结果字典
        result = {
            "Classification": {}
        }
        
        # 使用ViT模型预测分类结果
        image, transformed_image = self.preprocess_vit(image_path)
        
        # 检查分割模型是否检测到了龋齿(Caries)或深龋(Deep Caries)
        has_caries_detected = False
        if seg_results and "Location" in seg_results:
            if "Caries" in seg_results["Location"] or "Deep Caries" in seg_results["Location"]:
                has_caries_detected = True
        
        # 预测
        with torch.no_grad():
            transformed_image = transformed_image.unsqueeze(0).to(self.device)
            logits = self.model(transformed_image)[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # 对于每个类别，根据不同条件过滤
            for i, prob in enumerate(probs[0]):
                if prob > SCORE_THRESHOLD: # 首先必须大于0.3
                    disease = DISEASE_CATEGORIES[i]
                    threshold = SCORE_THRESHOLD
                    
                    # 根据不同疾病类型设置不同的阈值
                    if disease == "Alternation between primary and permanent teeth":
                        threshold = SCORE_THRESHOLD_Alternation
                    elif disease == "Impacted tooth":
                        threshold = SCORE_THRESHOLD_Impacted
                    elif disease == "Pulpitis":
                        threshold = SCORE_THRESHOLD_Pulpitis
                    elif disease == "Apical periodontitis":
                        threshold = SCORE_THRESHOLD_Apical_Periodontitis
                    elif disease == "Bone loss":
                        threshold = SCORE_THRESHOLD_Bone_loss
                    elif disease == "Caries":
                        threshold = SCORE_THRESHOLD_Caries
                    elif disease == "Periodontitis":
                        threshold = SCORE_THRESHOLD_Periodontitis
                    
                    # 如果是龋齿类别，并且分割模型没有检测到龋齿，则跳过
                    if disease == "Caries" and not has_caries_detected:
                        continue
                    
                    # 如果概率高于阈值，认为它存在
                    if prob > threshold:
                        result["Classification"][disease] = {
                            "present": True,
                            "score": float(prob)
                        }
        
        return result

    def _encode_mask(self, mask):
        """将二维掩码编码为轮廓"""
        contours = []
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        try:
            contours_cv, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_cv:
                contours.append(contour.reshape(-1).tolist())
        except:
            pass  # 如果无法找到轮廓，返回空列表
        
        return contours

def process_image(image_path, expert_6diseases, expert_7diseases, output_dir):
    """
    处理单个图像并生成或更新JSON
    
    Args:
        image_path: 图像文件路径
        expert_6diseases: 6diseases专家模型实例
        expert_7diseases: 7diseases专家模型实例
        output_dir: 输出目录
    
    Returns:
        处理的文件名
    """
    # 从路径获取文件名
    file_name = os.path.basename(image_path)
    image_id = str(os.path.splitext(file_name)[0])
    output_json_path = os.path.join(output_dir, f"{image_id}.json")
    
    with open(output_json_path, 'r') as f:
        existing_data = json.load(f)
    # 在现有数据基础上进行更改
    json_data = existing_data
    
    # 确保properties字段存在
    if "properties" not in json_data:
        json_data["properties"] = {}
    
    # 使用6diseases专家
    teeth_annotations = None
    if expert_6diseases:
        teeth_annotations = expert_6diseases.detect_6diseases(image_path)
        if teeth_annotations and "Location" in teeth_annotations:
            # 更新或添加Location字段
            json_data["properties"]["Location"] = teeth_annotations["Location"]
    
    # 使用7diseases专家
    if expert_7diseases:
        disease_annotations = expert_7diseases.detect_7diseases(image_path, teeth_annotations)
        if disease_annotations and "Classification" in disease_annotations:
            # 更新或添加Classification字段
            json_data["properties"]["Classification"] = disease_annotations["Classification"]
    
    # 将NumPy类型转换为标准Python类型，以便正确序列化到JSON
    json_data = convert_numpy_types(json_data)
    
    # 保存JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    return file_name

def main():
    # 设置路径
    image_dir = Path("/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-images")
    output_dir = Path("/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-jsons-0822")
    output_dir.mkdir(exist_ok=True)

    # 获取所有图像文件
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.PNG"))
    
    # 按数字排序
    sorted_files = sorted(image_files, key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)

    # 只取前500个
    # selected_files = sorted_files[:100]
    selected_files = sorted_files
    total_files = len(selected_files)
    
    print(f"Found {total_files} images to process")

    # 初始化两个专家模型
    print("Initializing 6diseases expert model...")
    expert_6diseases = DentalVisualExpert("6diseases")
    # expert_6diseases = None
    
    print("Initializing 7diseases expert model...")
    expert_7diseases = DentalVisualExpert("7diseases")
    
    # 创建进度条
    with tqdm(total=total_files, desc="Processing images") as pbar:
        # 处理每个图像
        for image_path in selected_files:
            try:
                # 处理图像
                file_name = process_image(image_path, expert_6diseases, expert_7diseases, output_dir)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"Current": file_name})
            
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                traceback.print_exc()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
