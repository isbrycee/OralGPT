import os
import json
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import datasets.transforms as T
import traceback
from tqdm import tqdm
import math
import argparse

import sys
import json

from main import build_model_main
from pycocotools import mask as mask_util
from util.slconfig import SLConfig
from util import box_ops

SCORE=0.3

def load_dino_model(model_type):
    if model_type == "teeth_id":
        model_config_path = "config/DINO/DINO_5scale_swinL_panoramic_x-ray_32ToothID.py"
        model_ckpt_path = "model_weights/Teeth_Visual_Experts_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.pth"
        category_map_path = "teeth_data/teeth_x-ray_teeth_id_numImages1598_ins_coco/annotations/updated_train.json"
    elif model_type == "4diseases":
        model_config_path = "config/DINO/1_disease_5scale_swin.py"
        model_ckpt_path = "model_weights/Teeth_Visual_Experts_dino_Swinl_x-ray_4diseases.pth"
        category_map_path = "teeth_data/teeth_x-ray_4diseases_numImages705_ins_coco/annotations/instances_train2017.json"
    elif model_type == "caries_filling":
        model_config_path = "config/DINO/3_caries_5scale_swin.py"
        model_ckpt_path = "model_weights/Teeth_Visual_Experts_dino_Swinl_x-ray_caries_filling.pth"
        category_map_path = "teeth_data/teeth_x-ray_caries_filling_numImages448_ins_coco/annotations/instances_train2017.json"
    elif model_type == "12diseases":
        model_config_path = "config/DINO/4_crown_5scale_swin.py"
        model_ckpt_path = "model_weights/Teeth_Visual_Experts_dino_Swinl_x-ray_12diseases.pth"
        category_map_path = "teeth_data/teeth_x-ray_12diseases_numImages9206_ins_coco/annotations/instances_train2017.json"
    elif model_type == "3periapical_lesions":
        model_config_path = "config/DINO/DINO_ r50_5scale_x-ray_periapical_lesions_3classes.py"
        model_ckpt_path = "model_weights/Teeth_Visual_Experts_DINO_r50_5scale_x-ray_periapical_lesions_3classes.pth"
        category_map_path = "teeth_data/teeth_x-ray_periapical_lesions_3classes_numImages3924_det_seg_coco/annotations_coco.json"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    args = SLConfig.fromfile(model_config_path)
    args.dataset_file = "coco"
    args.device = 'cuda'

    model, criterion, postprocessors = build_model_main(args)
    
    checkpoint = torch.load(model_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    if os.path.exists(category_map_path):
        with open(category_map_path) as f:
            data = json.load(f)
            if 'categories' in data:
                id2name = {cat['id']: cat['name'] for cat in data['categories']}
            else:
                id2name = {}
                for k, v in data.items():
                    try:
                        id2name[int(k)] = v
                    except ValueError:
                        continue
    else:
        id2name = None
    return model, postprocessors, id2name

def load_maskdino_model(model_type):
    sys.path.append("/hpc2hdd/home/yfan546/workplace/xray_teeth/MaskDINO")
    
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config
    from demo.predictor import VisualizationDemo
    
    if model_type == "quadrants":
        config_file = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/config/DINO/maskdino_SwinL_bs16_50ep_4s_dowsample1_2048_panoramic_x-ray_4quadrants.yaml"
        model_weights = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/model_weights/Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_4quadrants.pth"
        category_map_path = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/teeth_data/teeth_x-ray_1diseases_numImages7986_ins_seg_coco/instances_train2017.json"
    elif model_type == "bone_loss":
        config_file = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/config/DINO/maskdino_SwinL_bs16_50ep_4s_dowsample1_2048_x-ray_bone_loss_1diseases.yaml"
        model_weights = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/model_weights/Teeth_Visual_Experts_Maskdino_Swinl_x-ray_bone_loss_1disease.pth"
        category_map_path = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/teeth_data/teeth_x-ray_1diseases_numImages7986_ins_seg_coco/instances_train2017.json"
    elif model_type == "11diseases":
        config_file = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/config/DINO/maskdino_SwinL_bs16_50ep_4s_dowsample1_2048_x-ray_11diseases.yaml"
        model_weights = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/model_weights/Teeth_Visual_Experts_Maskdino_Swinl_x-ray_11diseases.pth"
        category_map_path = "/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/teeth_data/teeth_x-ray_11diseases_numImages8423_ins_seg_coco/annotations_coco.json"
    elif model_type == "mandibular":
        config_file = '/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/config/DINO/maskdino_Swinl_bs16_50ep_4s_dowsample1_2048_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.yaml'
        model_weights = '/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/model_weights/Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.pth.1'
        category_map_path = ''
    else:
        raise ValueError(f"Unknown MaskDINO model type: {model_type}")
    
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    
    predictor = VisualizationDemo(cfg)
    
    if os.path.exists(category_map_path):
        with open(category_map_path) as f:
            data = json.load(f)
            if 'categories' in data:
                id2name = {cat['id']: cat['name'] for cat in data['categories']}
            else:
                id2name = {}
                for k, v in data.items():
                    try:
                        id2name[int(k)] = v
                    except ValueError:
                        continue
    else:
        id2name = None
    
    return cfg, predictor, id2name

def convert_numpy_types(obj):
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

class DentalVisualExpert:
    def __init__(self, model_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.models = {}
        self.postprocessors = {}
        self.id2names = {}
        
        self.maskdino_models = {}
        self.maskdino_predictors = {}
        self.maskdino_id2names = {}
        
        # Load models based on type
        if model_type not in ["quadrants", "bone_loss", "11diseases", "mandibular"]:  
            model, postprocessor, id2name = load_dino_model(model_type)
            self.models[model_type] = model.to(self.device)
            self.postprocessors[model_type] = postprocessor
            self.id2names[model_type] = id2name
            print(f"Loaded {model_type} DINO model successfully")
        elif model_type in ["quadrants", "bone_loss", "11diseases", "mandibular"]:  
            cfg, predictor, id2name = load_maskdino_model(model_type)
            self.maskdino_models[model_type] = cfg
            self.maskdino_predictors[model_type] = predictor
            self.maskdino_id2names[model_type] = id2name
            print(f"Loaded {model_type} MaskDINO model successfully")
        
        # Image preprocessing transformation
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class mappings
        self.class_maps = {
            "teeth_id": {
                1: "11", 5: "12", 6: "13", 29: "14", 7: "15", 23: "16", 8: "17", 22: "18",
                2: "21", 3: "22", 4: "23", 27: "24", 28: "25", 30: "26", 9: "27", 10: "28",
                16: "31", 15: "32", 14: "33", 13: "34", 25: "35", 12: "36", 31: "37", 11: "38",
                17: "41", 24: "42", 18: "43", 19: "44", 20: "45", 32: "46", 26: "47", 21: "48"
            },
            "quadrants": {
                0: "Quadrant 2", 1: "Quadrant 1", 2: "Quadrant 3", 3: "Quadrant 4"
            },
            "4diseases": {
                0: "Impacted tooth", 1: "Caries", 2: "Periapical lesion", 3: "Deep caries"
            },
            "caries_filling": {
                1: "Filling", 2: "Caries"
            },
            "12diseases": {
                0: "Caries", 1: "Crown", 2: "Filling", 3: "Implant", 4: "Malaligned",
                5: "Mandibular canal", 6: "Missing teeth", 7: "Periapical lesions",
                8: "Retained root", 9: "Root canal treatment", 10: "Root piece", 11: "Impacted tooth"
            },
            "3periapical_lesions": {
                1: "Granuloma", 2: "Cyst", 3: "Abscess"
            },
            "bone_loss": { 0: "Bone loss"}
        }
        
        # Tooth ID mapping (FDI standard)
        self.tooth_ids = {
            # Upper right quadrant (Q1)
            "11": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "12": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "13": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "14": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "15": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "16": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "17": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": False},
            "18": {"quadrant": 1, "side": "upper right", "is_wisdom_tooth": True},
            
            # Upper left quadrant (Q2)
            "21": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "22": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "23": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "24": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "25": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "26": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "27": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": False},
            "28": {"quadrant": 2, "side": "upper left", "is_wisdom_tooth": True},
            
            # Lower left quadrant (Q3)
            "31": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "32": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "33": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "34": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "35": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "36": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "37": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": False},
            "38": {"quadrant": 3, "side": "lower left", "is_wisdom_tooth": True},
            
            # Lower right quadrant (Q4)
            "41": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "42": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "43": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "44": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "45": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "46": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "47": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": False},
            "48": {"quadrant": 4, "side": "lower right", "is_wisdom_tooth": True},
        }
        
        # Possible dental conditions
        self.conditions = [
            "Caries", "Deep caries", "Periapical lesions", "Impacted tooth",
            "Crown", "Filling", "Root canal treatment", "Implant",
            "Missing teeth", "Retained root", "Root piece", "Malaligned",
            "Mandibular canal", "Maxillary sinus", "Bone loss"
        ]

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transformed_image, _ = self.transform(image, None)
        return image, transformed_image

    def detect_teeth_id(self, image_path):
        orig_image, image_t = self.preprocess_image(image_path)
        orig_w, orig_h = orig_image.size
        
        teeth_model = self.models.get("teeth_id")
        teeth_annotations = []
        tooth_id_dict = {}
        
        if teeth_model:
            with torch.no_grad():
                outputs = teeth_model.cuda()(image_t[None].cuda())
                outputs = self.postprocessors.get("teeth_id")['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            
            for i in range(len(outputs['boxes'])):
                box_cxcywh = box_ops.box_xyxy_to_cxcywh(outputs['boxes'][i]).cpu().numpy()

                cx, cy, w, h = box_cxcywh
                box_orig = [
                    cx * orig_w,  
                    cy * orig_h,  
                    w * orig_w,   
                    h * orig_h    
                ]
                
                label = outputs['labels'][i].item()
                score = outputs['scores'][i].item()
                
                if score < SCORE:
                    continue
                    
                tooth_id = self.class_maps["teeth_id"].get(label, "unknown")
                
                id_annotation = {
                    "tooth_id": tooth_id,
                    "quadrant": self.tooth_ids.get(tooth_id, {}).get("quadrant", 0),
                    "is_wisdom_tooth": self.tooth_ids.get(tooth_id, {}).get("is_wisdom_tooth", False),
                    "side": self.tooth_ids.get(tooth_id, {}).get("side", "unknown"),
                    "bbox": np.round(self._convert_bbox_to_xyxy(box_orig)),
                    "center": np.round([box_orig[0], box_orig[1]]),
                    "score": round(score, 2),
                    "conditions": {}
                }
                
                # Keep only the highest scoring detection for each tooth_id
                if tooth_id in tooth_id_dict:
                    if score > tooth_id_dict[tooth_id]["score"]:
                        tooth_id_dict[tooth_id] = id_annotation
                else:
                    tooth_id_dict[tooth_id] = id_annotation
        
        teeth_annotations = list(tooth_id_dict.values())
        return teeth_annotations

    def detect_4diseases(self, image_path, json_data):
        orig_image, image_t = self.preprocess_image(image_path)
        orig_w, orig_h = orig_image.size
        
        disease_model = self.models.get("4diseases")
        if not disease_model:
            return json_data
            
        teeth_annotations = json_data["properties"].get("Teeth", [])
        
        with torch.no_grad():
            outputs = disease_model.cuda()(image_t[None].cuda())
            outputs = self.postprocessors.get("4diseases")['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            
        for i in range(len(outputs['boxes'])):
            box_xyxy = outputs['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box_xyxy
            disease_box_xyxy = [
                round(x1 * orig_w), 
                round(y1 * orig_h), 
                round(x2 * orig_w), 
                round(y2 * orig_h) 
            ]
            label = outputs['labels'][i].item()
            score = outputs['scores'][i].item()
            
            if score < SCORE:
                continue
                
            disease_name = self.class_maps["4diseases"].get(label)
            if not disease_name:
                continue
                
            if disease_name == "Caries":
                continue
                
            # Find tooth with maximum IoU with disease box
            closest_tooth = None
            max_iou = 0
            for tooth in teeth_annotations:
                tooth_box = tooth.get("bbox", [])
                if not tooth_box or len(tooth_box) != 4:
                    continue
                    
                tooth_box_xyxy = self._convert_bbox_to_xyxy(tooth_box)
                iou = self._calculate_iou_xyxy(disease_box_xyxy, tooth_box_xyxy)
                if iou > max_iou:
                    max_iou = iou
                    closest_tooth = tooth
                    
            # If found overlapping tooth, associate disease
            if max_iou > 0 and closest_tooth:
                if "conditions" not in closest_tooth:
                    closest_tooth["conditions"] = {}
                    
                if disease_name in closest_tooth["conditions"]:
                    existing_condition = closest_tooth["conditions"][disease_name]
                    if existing_condition.get("score", 0) < score:
                        closest_tooth["conditions"][disease_name] = {
                            "present": True,
                            "bbox": disease_box_xyxy,
                            "score": round(score, 2),
                            "iou": round(float(max_iou), 2)
                        }
                else:
                    closest_tooth["conditions"][disease_name] = {
                        "present": True,
                        "bbox": disease_box_xyxy,
                        "score": round(score, 2),
                        "iou": round(float(max_iou), 2)
                    }
            # If no matching tooth found
            elif max_iou == 0:
                unknown_tooth = None
                for tooth in teeth_annotations:
                    if tooth.get("tooth_id") == "unknown":
                        unknown_tooth = tooth
                        break
                        
                if unknown_tooth:
                    if "conditions" not in unknown_tooth:
                        unknown_tooth["conditions"] = {}
                    unknown_tooth["bbox"] = []
                    
                    if disease_name in unknown_tooth["conditions"]:
                        existing_condition = unknown_tooth["conditions"][disease_name]
                        if "bbox" in existing_condition:
                            if isinstance(existing_condition["bbox"], list):
                                if len(existing_condition["bbox"]) > 0 and isinstance(existing_condition["bbox"][0], list):
                                    overlap_found = False
                                    for i, existing_bbox in enumerate(existing_condition["bbox"]):
                                        existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_bbox)
                                        iou = self._calculate_iou_xyxy(disease_box_xyxy, existing_bbox_xyxy)
                                        if iou > 0:
                                            overlap_found = True
                                            if score > existing_condition.get("score", [0])[i]:
                                                existing_condition["bbox"][i] = disease_box_xyxy
                                                if isinstance(existing_condition["score"], list):
                                                    existing_condition["score"][i] = round(score, 2)
                                            break
                                    if not overlap_found:
                                        existing_condition["bbox"].append(disease_box_xyxy)
                                        if isinstance(existing_condition["score"], list):
                                            existing_condition["score"].append(round(score, 2))
                                        else:
                                            existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                else:
                                    existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_condition["bbox"])
                                    iou = self._calculate_iou_xyxy(disease_box_xyxy, existing_bbox_xyxy)
                                    if iou > 0:
                                        if score > existing_condition.get("score", 0):
                                            existing_condition["bbox"] = disease_box_xyxy
                                            existing_condition["score"] = round(score, 2)
                                    else:
                                        existing_condition["bbox"] = [existing_condition["bbox"], disease_box_xyxy]
                                        if isinstance(existing_condition["score"], (int, float)):
                                            existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                        else:
                                            existing_condition["score"].append(round(score, 2))
                            else:
                                existing_condition["bbox"] = disease_box_xyxy
                                existing_condition["score"] = round(score, 2)
                        else:
                            existing_condition["bbox"] = disease_box_xyxy
                            existing_condition["score"] = round(score, 2)
                    else:
                        unknown_tooth["conditions"][disease_name] = {
                            "present": True,
                            "bbox": disease_box_xyxy,
                            "score": round(score, 2)
                        }
                else:
                    unknown_tooth = {
                        "tooth_id": "unknown",
                        "quadrant": "unknown",
                        "is_wisdom_tooth": "unknown",
                        "side": "unknown",
                        "bbox": [],
                        "conditions": {
                            disease_name: {
                                "present": True,
                                "bbox": disease_box_xyxy,
                                "score": round(score, 2)
                            }
                        }
                    }
                    teeth_annotations.append(unknown_tooth)
        return json_data
   
    def detect_caries_filling(self, image_path, json_data):
        orig_image, image_t = self.preprocess_image(image_path)
        orig_w, orig_h = orig_image.size
        model = self.models.get("caries_filling")
        if not model:
            return json_data
        teeth_annotations = json_data["properties"].get("Teeth", [])
        
        with torch.no_grad():
            outputs = model.cuda()(image_t[None].cuda())
            outputs = self.postprocessors.get("caries_filling")['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        
        for i in range(len(outputs['boxes'])):
            box_xyxy = outputs['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box_xyxy
            condition_box_xyxy = [
                round(x1 * orig_w), 
                round(y1 * orig_h), 
                round(x2 * orig_w), 
                round(y2 * orig_h) 
            ]
            label = outputs['labels'][i].item()
            score = outputs['scores'][i].item()
            
            if score < SCORE:
                continue
                
            condition_name = self.class_maps["caries_filling"].get(label)
            if not condition_name:
                continue
                
            is_deep_caries = (condition_name == "Deep caries")
                
            closest_tooth = None
            max_iou = 0
            for tooth in teeth_annotations:
                tooth_box = tooth.get("bbox", [])
                if not tooth_box or len(tooth_box) != 4:
                    continue
                tooth_box_xyxy = self._convert_bbox_to_xyxy(tooth_box)
                iou = self._calculate_iou_xyxy(condition_box_xyxy, tooth_box_xyxy)
                if iou > max_iou:
                    max_iou = iou
                    closest_tooth = tooth
                    
            if max_iou > 0 and closest_tooth:
                if "conditions" not in closest_tooth:
                    closest_tooth["conditions"] = {}
                    
                if is_deep_caries:
                    if "Caries" in closest_tooth["conditions"]:
                        caries_condition = closest_tooth["conditions"]["Caries"]
                        caries_bbox = caries_condition.get("bbox", [])
                        if caries_bbox:
                            caries_bbox_xyxy = self._convert_bbox_to_xyxy(caries_bbox)
                            iou_with_caries = self._calculate_iou_xyxy(condition_box_xyxy, caries_bbox_xyxy)
                            if iou_with_caries > 0:
                                closest_tooth["conditions"]["Deep caries"] = {
                                    "present": True,
                                    "bbox": condition_box_xyxy,
                                    "score": round(score, 2),
                                    "iou": round(float(max_iou), 2)
                                }
                                del closest_tooth["conditions"]["Caries"]
                                continue
                    
                    if "Deep caries" in closest_tooth["conditions"]:
                        deep_caries_condition = closest_tooth["conditions"]["Deep caries"]
                        deep_caries_bbox = deep_caries_condition.get("bbox", [])
                        if deep_caries_bbox:
                            deep_caries_bbox_xyxy = self._convert_bbox_to_xyxy(deep_caries_bbox)
                            iou_with_deep_caries = self._calculate_iou_xyxy(condition_box_xyxy, deep_caries_bbox_xyxy)
                            if iou_with_deep_caries > 0:
                                if score > deep_caries_condition.get("score", 0):
                                    closest_tooth["conditions"]["Deep caries"] = {
                                        "present": True,
                                        "bbox": condition_box_xyxy,
                                        "score": round(score, 2),
                                        "iou": round(float(max_iou), 2)
                                    }
                                continue
                
                elif condition_name == "Caries":
                    if "Deep caries" in closest_tooth["conditions"]:
                        deep_caries_condition = closest_tooth["conditions"]["Deep caries"]
                        deep_caries_bbox = deep_caries_condition.get("bbox", [])
                        if deep_caries_bbox:
                            deep_caries_bbox_xyxy = self._convert_bbox_to_xyxy(deep_caries_bbox)
                            iou_with_deep_caries = self._calculate_iou_xyxy(condition_box_xyxy, deep_caries_bbox_xyxy)
                            if iou_with_deep_caries > 0:
                                continue
                
                if condition_name in closest_tooth["conditions"]:
                    existing_condition = closest_tooth["conditions"][condition_name]
                    if existing_condition.get("score", 0) < score:
                        closest_tooth["conditions"][condition_name] = {
                            "present": True,
                            "bbox": condition_box_xyxy,
                            "score": round(score, 2),
                            "iou": round(float(max_iou), 2)
                        }
                else:
                    closest_tooth["conditions"][condition_name] = {
                        "present": True,
                        "bbox": condition_box_xyxy,
                        "score": round(score, 2),
                        "iou": round(float(max_iou), 2)
                    }
            elif max_iou == 0:
                unknown_tooth = None
                for tooth in teeth_annotations:
                    if tooth.get("tooth_id") == "unknown":
                        unknown_tooth = tooth
                        break
                        
                if unknown_tooth:
                    if "conditions" not in unknown_tooth:
                        unknown_tooth["conditions"] = {}
                    unknown_tooth["bbox"] = []
                    
                    if is_deep_caries or condition_name == "Caries":
                        caries_exists = "Caries" in unknown_tooth["conditions"]
                        deep_caries_exists = "Deep caries" in unknown_tooth["conditions"]
                        
                        if is_deep_caries and caries_exists:
                            caries_condition = unknown_tooth["conditions"]["Caries"]
                            caries_bbox = caries_condition.get("bbox", [])
                            
                            if isinstance(caries_bbox, list) and not isinstance(caries_bbox[0], list):
                                caries_bbox_xyxy = self._convert_bbox_to_xyxy(caries_bbox)
                                iou_with_caries = self._calculate_iou_xyxy(condition_box_xyxy, caries_bbox_xyxy)
                                
                                if iou_with_caries > 0:
                                    unknown_tooth["conditions"]["Deep caries"] = {
                                        "present": True,
                                        "bbox": condition_box_xyxy,
                                        "score": round(score, 2)
                                    }
                                    del unknown_tooth["conditions"]["Caries"]
                                    continue
                        
                        if condition_name == "Caries" and deep_caries_exists:
                            deep_caries_condition = unknown_tooth["conditions"]["Deep caries"]
                            deep_caries_bbox = deep_caries_condition.get("bbox", [])
                            
                            if isinstance(deep_caries_bbox, list) and not isinstance(deep_caries_bbox[0], list):
                                deep_caries_bbox_xyxy = self._convert_bbox_to_xyxy(deep_caries_bbox)
                                iou_with_deep_caries = self._calculate_iou_xyxy(condition_box_xyxy, deep_caries_bbox_xyxy)
                                
                                if iou_with_deep_caries > 0:
                                    continue
                    
                    if condition_name in unknown_tooth["conditions"]:
                        existing_condition = unknown_tooth["conditions"][condition_name]
                        if "bbox" in existing_condition:
                            if isinstance(existing_condition["bbox"], list):
                                if len(existing_condition["bbox"]) > 0 and isinstance(existing_condition["bbox"][0], list):
                                    overlap_found = False
                                    for i, existing_bbox in enumerate(existing_condition["bbox"]):
                                        existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_bbox)
                                        iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                                        if iou > 0:
                                            overlap_found = True
                                            if score > existing_condition.get("score", [0])[i]:
                                                existing_condition["bbox"][i] = condition_box_xyxy
                                                if isinstance(existing_condition["score"], list):
                                                    existing_condition["score"][i] = round(score, 2)
                                            break
                                    if not overlap_found:
                                        existing_condition["bbox"].append(condition_box_xyxy)
                                        if isinstance(existing_condition["score"], list):
                                            existing_condition["score"].append(round(score, 2))
                                        else:
                                            existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                else:
                                    existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_condition["bbox"])
                                    iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                                    if iou > 0:
                                        if score > existing_condition.get("score", 0):
                                            existing_condition["bbox"] = condition_box_xyxy
                                            existing_condition["score"] = round(score, 2)
                                    else:
                                        existing_condition["bbox"] = [existing_condition["bbox"], condition_box_xyxy]
                                        if isinstance(existing_condition["score"], (int, float)):
                                            existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                        else:
                                            existing_condition["score"].append(round(score, 2))
                            else:
                                existing_condition["bbox"] = condition_box_xyxy
                                existing_condition["score"] = round(score, 2)
                        else:
                            existing_condition["bbox"] = condition_box_xyxy
                            existing_condition["score"] = round(score, 2)
                    else:
                        unknown_tooth["conditions"][condition_name] = {
                            "present": True,
                            "bbox": condition_box_xyxy,
                            "score": round(score, 2)
                        }
                else:
                    unknown_tooth = {
                        "tooth_id": "unknown",
                        "quadrant": "unknown",
                        "is_wisdom_tooth": "unknown",
                        "side": "unknown",
                        "bbox": [],
                        "conditions": {
                            condition_name: {
                                "present": True,
                                "bbox": condition_box_xyxy,
                                "score": round(score, 2)
                            }
                        }
                    }
                    teeth_annotations.append(unknown_tooth)
        return json_data

    def detect_12diseases(self, image_path, json_data):
        orig_image, image_t = self.preprocess_image(image_path)
        orig_w, orig_h = orig_image.size
        disease_model = self.models.get("12diseases")
        if not disease_model:
            return json_data
        teeth_annotations = json_data["properties"].get("Teeth", [])
        
        if "Missing teeth" not in json_data["properties"]:
            json_data["properties"]["Missing teeth"] = []
            
        teeth_areas = []
        for tooth in teeth_annotations:
            tooth_box = tooth.get("bbox", [])
            if not tooth_box or len(tooth_box) != 4:
                continue
            tooth_box_xyxy = self._convert_bbox_to_xyxy(tooth_box)
            tooth_area = (tooth_box_xyxy[2] - tooth_box_xyxy[0]) * (tooth_box_xyxy[3] - tooth_box_xyxy[1])
            teeth_areas.append(tooth_area)
            
        if not teeth_areas:
            avg_tooth_area = float('inf')
        else:
            avg_tooth_area = sum(teeth_areas) / len(teeth_areas)
            
        with torch.no_grad():
            outputs = disease_model.cuda()(image_t[None].cuda())
            outputs = self.postprocessors.get("12diseases")['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            
        class_map = {
            0: "Caries", 1: "Crown", 2: "Filling", 3: "Implant", 4: "Malaligned",
            5: "Mandibular canal", 6: "Missing teeth", 7: "Periapical lesions",
            8: "Retained root", 9: "Root canal treatment", 10: "Root piece", 11: "Impacted tooth"
        }
        
        for i in range(len(outputs['boxes'])):
            box_xyxy = outputs['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box_xyxy
            condition_box_xyxy = [
                round(x1 * orig_w), 
                round(y1 * orig_h), 
                round(x2 * orig_w), 
                round(y2 * orig_h) 
            ]
            label = outputs['labels'][i].item()
            score = outputs['scores'][i].item()
            
            if score < SCORE:
                continue
                
            condition_name = class_map.get(label, "unknown")
            if condition_name == "unknown":
                continue
                
            if condition_name == "Missing teeth":
                missing_teeth = json_data["properties"]["Missing teeth"]
                overlap_found = False
                
                for j, existing_missing in enumerate(missing_teeth):
                    existing_bbox = existing_missing.get("bbox", [])
                    if not existing_bbox:
                        continue
                        
                    existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_bbox)
                    iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                    
                    if iou > 0:
                        overlap_found = True
                        if score > existing_missing.get("score", 0):
                            missing_teeth[j] = {
                                "bbox": condition_box_xyxy,
                                "score": round(score, 2)
                            }
                        break
                
                if not overlap_found:
                    missing_teeth.append({
                        "bbox": condition_box_xyxy,
                        "score": round(score, 2)
                    })
                
                continue
                
            if condition_name == "Mandibular canal":
                if "JawBones" not in json_data["properties"]:
                    json_data["properties"]["JawBones"] = [{"conditions": {}}]
                elif len(json_data["properties"]["JawBones"]) == 0:
                    json_data["properties"]["JawBones"].append({"conditions": {}})
                    
                if "conditions" not in json_data["properties"]["JawBones"][0]:
                    json_data["properties"]["JawBones"][0]["conditions"] = {}
                    
                json_data["properties"]["JawBones"][0]["conditions"]["Mandibular canal"] = {
                    "bbox": condition_box_xyxy,
                    "segmentation": [],
                    "score": round(score, 2)
                }
                continue
                
            if condition_name == "Periapical lesions":
                standardized_name = "Periapical lesions"
                lesion_area = (condition_box_xyxy[2] - condition_box_xyxy[0]) * (condition_box_xyxy[3] - condition_box_xyxy[1])
                if lesion_area > avg_tooth_area:
                    continue
            elif condition_name == "Impacted tooth":
                standardized_name = "Impacted tooth"
            else:
                standardized_name = condition_name
                
            closest_tooth = None
            max_iou = 0
            for tooth in teeth_annotations:
                tooth_box = tooth.get("bbox", [])
                if not tooth_box or len(tooth_box) != 4:
                    continue
                tooth_box_xyxy = self._convert_bbox_to_xyxy(tooth_box)
                iou = self._calculate_iou_xyxy(condition_box_xyxy, tooth_box_xyxy)
                if iou > max_iou:
                    max_iou = iou
                    closest_tooth = tooth
                    
            if max_iou > 0 and closest_tooth:
                if "conditions" not in closest_tooth:
                    closest_tooth["conditions"] = {}
                    
                if standardized_name in closest_tooth["conditions"]:
                    existing_condition = closest_tooth["conditions"][standardized_name]
                    if existing_condition.get("score", 0) < score:
                        closest_tooth["conditions"][standardized_name] = {
                            "present": True,
                            "bbox": condition_box_xyxy,
                            "score": round(score, 2),
                            "iou": round(float(max_iou), 2)
                        }
                else:
                    closest_tooth["conditions"][standardized_name] = {
                        "present": True,
                        "bbox": condition_box_xyxy,
                        "score": round(score, 2),
                        "iou": round(float(max_iou), 2)
                    }
            elif max_iou == 0:
                unknown_tooth = None
                for tooth in teeth_annotations:
                    if tooth.get("tooth_id") == "unknown":
                        unknown_tooth = tooth
                        break
                        
                if unknown_tooth:
                    if "conditions" not in unknown_tooth:
                        unknown_tooth["conditions"] = {}
                    unknown_tooth["bbox"] = []
                    
                    if standardized_name in unknown_tooth["conditions"]:
                        existing_condition = unknown_tooth["conditions"][standardized_name]
                        if "bbox" in existing_condition:
                            if isinstance(existing_condition["bbox"], list):
                                if len(existing_condition["bbox"]) > 0 and isinstance(existing_condition["bbox"][0], list):
                                    overlap_found = False
                                    for i, existing_bbox in enumerate(existing_condition["bbox"]):
                                        existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_bbox)
                                        iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                                        if iou > 0:
                                            overlap_found = True
                                            if score > existing_condition.get("score", [0])[i]:
                                                existing_condition["bbox"][i] = condition_box_xyxy
                                                if isinstance(existing_condition["score"], list):
                                                    existing_condition["score"][i] = round(score, 2)
                                            break
                                    if not overlap_found:
                                        existing_condition["bbox"].append(condition_box_xyxy)
                                        if isinstance(existing_condition["score"], list):
                                            existing_condition["score"].append(round(score, 2))
                                        else:
                                            existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                else:
                                    existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_condition["bbox"])
                                    iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                                    if iou > 0:
                                        if score > existing_condition.get("score", 0):
                                            existing_condition["bbox"] = condition_box_xyxy
                                            existing_condition["score"] = round(score, 2)
                                    else:
                                        existing_condition["bbox"] = [existing_condition["bbox"], condition_box_xyxy]
                                        if isinstance(existing_condition["score"], (int, float)):
                                            existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                        else:
                                            existing_condition["score"].append(round(score, 2))
                            else:
                                existing_condition["bbox"] = condition_box_xyxy
                                existing_condition["score"] = round(score, 2)
                        else:
                            existing_condition["bbox"] = condition_box_xyxy
                            existing_condition["score"] = round(score, 2)
                    else:
                        unknown_tooth["conditions"][standardized_name] = {
                            "present": True,
                            "bbox": condition_box_xyxy,
                            "score": round(score, 2)
                        }
                else:
                    unknown_tooth = {
                        "tooth_id": "unknown",
                        "quadrant": "unknown",
                        "is_wisdom_tooth": "unknown",
                        "side": "unknown",
                        "bbox": [],
                        "conditions": {
                            standardized_name: {
                                "present": True,
                                "bbox": condition_box_xyxy,
                                "score": round(score, 2)
                            }
                        }
                    }
                    teeth_annotations.append(unknown_tooth)
        return json_data

    def detect_3periapical_lesions(self, image_path, json_data):
        orig_image, image_t = self.preprocess_image(image_path)
        orig_w, orig_h = orig_image.size
        model = self.models.get("3periapical_lesions")
        if not model:
            return json_data
        teeth_annotations = json_data["properties"].get("Teeth", [])
        
        teeth_areas = []
        for tooth in teeth_annotations:
            tooth_box = tooth.get("bbox", [])
            if not tooth_box or len(tooth_box) != 4:
                continue
            tooth_box_xyxy = self._convert_bbox_to_xyxy(tooth_box)
            tooth_area = (tooth_box_xyxy[2] - tooth_box_xyxy[0]) * (tooth_box_xyxy[3] - tooth_box_xyxy[1])
            teeth_areas.append(tooth_area)
            
        if not teeth_areas:
            avg_tooth_area = float('inf')
        else:
            avg_tooth_area = sum(teeth_areas) / len(teeth_areas)
            
        with torch.no_grad():
            outputs = model.cuda()(image_t[None].cuda())
            outputs = self.postprocessors.get("3periapical_lesions")['bbox'](outputs, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            
        for i in range(len(outputs['boxes'])):
            box_xyxy = outputs['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = box_xyxy
            condition_box_xyxy = [
                round(x1 * orig_w), 
                round(y1 * orig_h), 
                round(x2 * orig_w), 
                round(y2 * orig_h) 
            ]
            
            lesion_area = (condition_box_xyxy[2] - condition_box_xyxy[0]) * (condition_box_xyxy[3] - condition_box_xyxy[1])
            if lesion_area > avg_tooth_area:
                continue
                
            label = outputs['labels'][i].item()
            score = outputs['scores'][i].item()
            
            if score < SCORE:
                continue
                
            condition_name = self.class_maps["3periapical_lesions"].get(label)
            if not condition_name:
                continue
                
            main_condition = "Periapical lesions"
            
            closest_tooth = None
            max_iou = 0
            for tooth in teeth_annotations:
                tooth_box = tooth.get("bbox", [])
                if not tooth_box or len(tooth_box) != 4:
                    continue
                tooth_box_xyxy = self._convert_bbox_to_xyxy(tooth_box)
                iou = self._calculate_iou_xyxy(condition_box_xyxy, tooth_box_xyxy)
                if iou > max_iou:
                    max_iou = iou
                    closest_tooth = tooth
                    
            if max_iou > 0 and closest_tooth:
                if "conditions" not in closest_tooth:
                    closest_tooth["conditions"] = {}
                    
                if main_condition not in closest_tooth["conditions"] or closest_tooth["conditions"][main_condition].get("score", 0) < score:
                    closest_tooth["conditions"][main_condition] = {
                        "present": True,
                        "bbox": condition_box_xyxy,
                        "score": round(score, 2),
                        "type": condition_name,
                        "iou": round(float(max_iou), 2)
                    }
            elif max_iou == 0:
                unknown_tooth = None
                for tooth in teeth_annotations:
                    if tooth.get("tooth_id") == "unknown":
                        unknown_tooth = tooth
                        break
                        
                if unknown_tooth:
                    if "conditions" not in unknown_tooth:
                        unknown_tooth["conditions"] = {}
                        
                    if main_condition in unknown_tooth["conditions"]:
                        existing_condition = unknown_tooth["conditions"][main_condition]
                        if isinstance(existing_condition.get("bbox", []), list) and isinstance(existing_condition.get("score", []), list):
                            existing_bboxes = existing_condition.get("bbox", [])
                            existing_scores = existing_condition.get("score", [])
                            existing_types = existing_condition.get("type", [])
                            
                            has_overlap = False
                            replace_index = -1
                            for j, existing_bbox in enumerate(existing_bboxes):
                                existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_bbox)
                                overlap_iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                                if overlap_iou > 0:
                                    has_overlap = True
                                    if score > existing_scores[j]:
                                        replace_index = j
                                    break
                                    
                            if has_overlap and replace_index >= 0:
                                existing_bboxes[replace_index] = condition_box_xyxy
                                existing_scores[replace_index] = round(score, 2)
                                existing_types[replace_index] = condition_name
                            elif not has_overlap:
                                existing_bboxes.append(condition_box_xyxy)
                                existing_scores.append(round(score, 2))
                                existing_types.append(condition_name)
                                
                            unknown_tooth["conditions"][main_condition] = {
                                "present": True,
                                "bbox": existing_bboxes,
                                "score": existing_scores,
                                "type": existing_types
                            }
                        else:
                            existing_bbox = existing_condition.get("bbox", [])
                            existing_score = existing_condition.get("score", 0)
                            existing_type = existing_condition.get("type", "")
                            
                            existing_bbox_xyxy = self._convert_bbox_to_xyxy(existing_bbox)
                            
                            overlap_iou = self._calculate_iou_xyxy(condition_box_xyxy, existing_bbox_xyxy)
                            if overlap_iou > 0:
                                if score > existing_score:
                                    unknown_tooth["conditions"][main_condition] = {
                                        "present": True,
                                        "bbox": [condition_box_xyxy],
                                        "score": [round(score, 2)],
                                        "type": [condition_name]
                                    }
                                else:
                                    unknown_tooth["conditions"][main_condition] = {
                                        "present": True,
                                        "bbox": [existing_bbox_xyxy],
                                        "score": [existing_score],
                                        "type": [existing_type]
                                    }
                            else:
                                unknown_tooth["conditions"][main_condition] = {
                                    "present": True,
                                    "bbox": [existing_bbox_xyxy, condition_box_xyxy],
                                    "score": [existing_score, round(score, 2)],
                                    "type": [existing_type, condition_name]
                                }
                    else:
                        unknown_tooth["conditions"][main_condition] = {
                            "present": True,
                            "bbox": [condition_box_xyxy],
                            "score": [round(score, 2)],
                            "type": [condition_name]
                        }
                else:
                    unknown_tooth = {
                        "tooth_id": "unknown",
                        "quadrant": "unknown",
                        "is_wisdom_tooth": "unknown",
                        "side": "unknown",
                        "bbox": [],
                        "conditions": {
                            main_condition: {
                                "present": True,
                                "bbox": [condition_box_xyxy],
                                "score": [round(score, 2)],
                                "type": [condition_name]
                            }
                        }
                    }
                    teeth_annotations.append(unknown_tooth)
        return json_data

    def detect_quadrants_with_maskdino(self, image_path):
        predictor = self.maskdino_predictors.get("quadrants")
        if not predictor:
            return []
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return []
            
        predictions, _ = predictor.run_on_image(img)
        quadrants_annotations = []
        
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
                scores = instances.scores.numpy() if instances.has("scores") else []
                class_ids = instances.pred_classes.numpy() if instances.has("pred_classes") else []
                masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
                
                high_conf_indices = np.where(scores > SCORE)[0]
                for idx in high_conf_indices:
                    box = boxes[idx]
                    score = float(scores[idx])
                    class_id = int(class_ids[idx])
                    mask = masks[idx] if len(masks) > idx else None
                    
                    quadrant_name = self.class_maps["quadrants"].get(class_id)
                    if not quadrant_name:
                        continue
                        
                    x1 = round(float(box[0]))
                    y1 = round(float(box[1]))
                    x2 = round(float(box[2]))
                    y2 = round(float(box[3]))
                    box_xyxy = [x1, y1, x2, y2]
                    
                    segmentation = None
                    if mask is not None:
                        segmentation = self._encode_mask(mask)
                        if segmentation:
                            segmentation = {
                                "size": segmentation["size"],
                                "counts": segmentation["counts"]
                            }
                    
                    quadrant_annotation = {
                        "quadrant": quadrant_name,
                        "present": True,
                        "bbox": box_xyxy,
                        "segmentation": segmentation,
                        "score": round(score, 2)
                    }
                    quadrants_annotations.append(quadrant_annotation)
        return quadrants_annotations

    def detect_11diseases(self, image_path, json_data):
        predictor = self.maskdino_predictors.get("11diseases")
        if not predictor:
            return json_data
            
        teeth_annotations = json_data["properties"].get("Teeth", [])
        
        teeth_areas = []
        for tooth in teeth_annotations:
            tooth_box = tooth.get("bbox", [])
            if not tooth_box or len(tooth_box) != 4:
                continue
            if len(tooth_box) == 4:
                tooth_area = (tooth_box[2] - tooth_box[0]) * (tooth_box[3] - tooth_box[1])
                teeth_areas.append(tooth_area)
                
        if not teeth_areas:
            avg_tooth_area = float('inf')
        else:
            avg_tooth_area = sum(teeth_areas) / len(teeth_areas)
        
        disease_map = {
            1: "Caries", 2: "Crown", 3: "Filling", 4: "Implant", 5: "Mandibular Canal",
            6: "Missing teeth", 7: "Periapical lesion", 8: "Root Canal Treatment",
            9: "Root Piece", 10: "Impacted tooth", 11: "Maxillary sinus"
        }

        standardized_names = {
            "Caries": "Caries", "Crown": "Crown", "Filling": "Filling", "Implant": "Implant",
            "Mandibular Canal": "Mandibular canal", "Missing teeth": "Missing teeth",
            "Periapical lesion": "Periapical lesions", "Root Canal Treatment": "Root canal treatment",
            "Root Piece": "Root piece", "Impacted tooth": "Impacted tooth", 
            "Maxillary sinus": "Maxillary sinus"
        }
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return json_data
        
        try:
            predictions, _ = predictor.run_on_image(img)
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                if len(instances) > 0:
                    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
                    scores = instances.scores.numpy() if instances.has("scores") else []
                    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
                    class_ids = instances.pred_classes.numpy() if instances.has("pred_classes") else []
                    
                    high_conf_indices = np.where(scores > SCORE)[0]
                    for idx in high_conf_indices:
                        if idx >= len(boxes) or idx >= len(scores) or (len(masks) > 0 and idx >= len(masks)) or idx >= len(class_ids):
                            continue
                        
                        box = boxes[idx]
                        score = float(scores[idx])
                        mask = masks[idx] if len(masks) > idx else None
                        class_id = int(class_ids[idx]) + 1
                        
                        condition_name = disease_map.get(class_id, "unknown")
                        if condition_name == "unknown":
                            continue
                        
                        x1 = round(float(box[0]))
                        y1 = round(float(box[1]))
                        x2 = round(float(box[2]))
                        y2 = round(float(box[3]))
                        box_xyxy = [x1, y1, x2, y2]
                        
                        std_name = standardized_names.get(condition_name, condition_name)
                        
                        if std_name == "Periapical lesions":
                            lesion_area = (x2 - x1) * (y2 - y1)
                            if lesion_area > avg_tooth_area:
                                continue
                        
                        segmentation = None
                        if mask is not None:
                            segmentation = self._encode_mask(mask)
                            if segmentation:
                                segmentation = {
                                    "size": segmentation["size"],
                                    "counts": segmentation["counts"]
                                }
                        
                        if std_name == "Missing teeth":
                            if "MissingTeeth" not in json_data["properties"]:
                                json_data["properties"]["Missing Teeth"] = []
                            
                            existing_missing_teeth = json_data["properties"]["Missing Teeth"]
                            overlap_found = False
                            
                            for i, existing_teeth in enumerate(existing_missing_teeth):
                                existing_box = existing_teeth.get("bbox", [])
                                existing_score = existing_teeth.get("score", 0)
                                
                                if existing_box:
                                    iou = self._calculate_iou_xyxy(box_xyxy, existing_box)
                                    if iou > 0:
                                        overlap_found = True
                                        if score > existing_score:
                                            existing_missing_teeth[i] = {
                                                "bbox": box_xyxy,
                                                "segmentation": segmentation,
                                                "score": round(score, 2)
                                            }
                                        break
                            
                            if not overlap_found:
                                json_data["properties"]["Missing Teeth"].append({
                                    "bbox": box_xyxy,
                                    "segmentation": segmentation,
                                    "score": round(score, 2)
                                })
                            
                            continue
                        
                        if condition_name in ["Mandibular Canal", "Maxillary sinus"]:
                            std_name = standardized_names.get(condition_name)
                            if not std_name:
                                continue
                            
                            if "JawBones" not in json_data["properties"]:
                                json_data["properties"]["JawBones"] = [{"conditions": {}}]
                            elif len(json_data["properties"]["JawBones"]) == 0:
                                json_data["properties"]["JawBones"].append({"conditions": {}})
                            
                            if "conditions" not in json_data["properties"]["JawBones"][0]:
                                json_data["properties"]["JawBones"][0]["conditions"] = {}
                            
                            json_data["properties"]["JawBones"][0]["conditions"][std_name] = {
                                "bbox": box_xyxy,
                                "segmentation": segmentation,
                                "score": round(score, 2)
                            }
                            continue
                        
                        closest_tooth = None
                        max_iou = 0
                        for tooth in teeth_annotations:
                            tooth_box = tooth.get("bbox", [])
                            if not tooth_box or len(tooth_box) != 4:
                                continue
                            
                            iou = self._calculate_iou_xyxy(box_xyxy, tooth_box)
                            if iou > max_iou:
                                max_iou = iou
                                closest_tooth = tooth
                        
                        if max_iou > 0 and closest_tooth:
                            if "conditions" not in closest_tooth:
                                closest_tooth["conditions"] = {}
                            
                            if std_name in closest_tooth["conditions"]:
                                existing_condition = closest_tooth["conditions"][std_name]
                                if existing_condition.get("score", 0) < score:
                                    closest_tooth["conditions"][std_name] = {
                                        "present": True,
                                        "bbox": box_xyxy,
                                        "segmentation": segmentation,
                                        "score": round(score, 2),
                                        "iou": round(float(max_iou), 2)
                                    }
                            else:
                                closest_tooth["conditions"][std_name] = {
                                    "present": True,
                                    "bbox": box_xyxy,
                                    "segmentation": segmentation,
                                    "score": round(score, 2),
                                    "iou": round(float(max_iou), 2)
                                }
                        elif max_iou == 0:
                            unknown_tooth = None
                            for tooth in teeth_annotations:
                                if tooth.get("tooth_id") == "unknown":
                                    unknown_tooth = tooth
                                    break
                            
                            if unknown_tooth:
                                if "conditions" not in unknown_tooth:
                                    unknown_tooth["conditions"] = {}
                                
                                unknown_tooth["bbox"] = []
                                
                                if std_name in unknown_tooth["conditions"]:
                                    existing_condition = unknown_tooth["conditions"][std_name]
                                    
                                    # 处理未知牙齿的情况 - 特别处理
                                    if "bbox" in existing_condition:
                                        # 确保bbox是列表格式
                                        if not isinstance(existing_condition["bbox"], list):
                                            existing_condition["bbox"] = [existing_condition["bbox"]]
                                            if "segmentation" in existing_condition:
                                                existing_condition["segmentation"] = [existing_condition["segmentation"]]
                                            else:
                                                existing_condition["segmentation"] = [None]
                                            if isinstance(existing_condition["score"], (int, float)):
                                                existing_condition["score"] = [existing_condition["score"]]
                                        
                                        # 检查bbox列表元素的类型
                                        if len(existing_condition["bbox"]) > 0:
                                            if not isinstance(existing_condition["bbox"][0], list):
                                                # 如果第一个元素不是列表，转换为嵌套列表
                                                existing_condition["bbox"] = [existing_condition["bbox"]]
                                                if "segmentation" in existing_condition:
                                                    existing_condition["segmentation"] = [existing_condition["segmentation"]]
                                                else:
                                                    existing_condition["segmentation"] = [[None]]
                                                if isinstance(existing_condition["score"], (int, float)):
                                                    existing_condition["score"] = [existing_condition["score"]]
                                        
                                        # 从这里开始处理带有嵌套列表的bbox
                                        overlap_found = False
                                        for i, existing_bbox in enumerate(existing_condition["bbox"]):
                                            iou = self._calculate_iou_xyxy(box_xyxy, existing_bbox)
                                            if iou > 0:
                                                overlap_found = True
                                                # 检查score是否为列表并确保长度足够
                                                if not isinstance(existing_condition["score"], list):
                                                    existing_condition["score"] = [existing_condition["score"]]
                                                elif len(existing_condition["score"]) <= i:
                                                    # 扩展score列表
                                                    existing_condition["score"].extend([0] * (i + 1 - len(existing_condition["score"])))
                                                
                                                if score > existing_condition["score"][i]:
                                                    existing_condition["bbox"][i] = box_xyxy
                                                    
                                                    # 确保segmentation存在并且有足够的元素
                                                    if "segmentation" not in existing_condition:
                                                        existing_condition["segmentation"] = [None] * len(existing_condition["bbox"])
                                                    elif len(existing_condition["segmentation"]) <= i:
                                                        # 扩展segmentation列表
                                                        existing_condition["segmentation"].extend([None] * (len(existing_condition["bbox"]) - len(existing_condition["segmentation"])))
                                                    
                                                    existing_condition["segmentation"][i] = segmentation
                                                    existing_condition["score"][i] = round(score, 2)
                                                break
                                        
                                        if not overlap_found:
                                            existing_condition["bbox"].append(box_xyxy)
                                            
                                            # 确保segmentation和score列表正确扩展
                                            if "segmentation" not in existing_condition:
                                                existing_condition["segmentation"] = [None] * (len(existing_condition["bbox"]) - 1)
                                                existing_condition["segmentation"].append(segmentation)
                                            else:
                                                existing_condition["segmentation"].append(segmentation)
                                            
                                            if not isinstance(existing_condition["score"], list):
                                                existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                            else:
                                                existing_condition["score"].append(round(score, 2))
                                    else:
                                        # 新建条件项
                                        existing_condition["bbox"] = [box_xyxy]
                                        existing_condition["segmentation"] = [segmentation]
                                        existing_condition["score"] = [round(score, 2)]
                                        existing_condition["present"] = True
                                else:
                                    # 创建新条件
                                    unknown_tooth["conditions"][std_name] = {
                                        "present": True,
                                        "bbox": [box_xyxy],
                                        "segmentation": [segmentation],
                                        "score": [round(score, 2)]
                                    }
                            else:
                                # 创建新的未知牙齿
                                unknown_tooth = {
                                    "tooth_id": "unknown",
                                    "quadrant": "unknown",
                                    "is_wisdom_tooth": "unknown",
                                    "side": "unknown",
                                    "bbox": [],
                                    "conditions": {
                                        std_name: {
                                            "present": True,
                                            "bbox": [box_xyxy],
                                            "segmentation": [segmentation],
                                            "score": [round(score, 2)]
                                        }
                                    }
                                }
                                teeth_annotations.append(unknown_tooth)
        except Exception as e:
            print(f"Error in detect_11diseases for {image_path}: {e}")
            traceback.print_exc()
        return json_data
    
    def detect_bone_loss(self, image_path, json_data):
        predictor = self.maskdino_predictors.get("bone_loss")
        if not predictor:
            return json_data
            
        img = cv2.imread(image_path)
        predictions, _ = predictor.run_on_image(img)
        
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
                scores = instances.scores.numpy() if instances.has("scores") else []
                masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
                
                high_conf_indices = np.where(scores > SCORE)[0]
                if "JawBones" not in json_data["properties"]:
                    json_data["properties"]["JawBones"] = []
                
                for idx in high_conf_indices:
                    box = boxes[idx]
                    score = float(scores[idx])
                    mask = masks[idx] if len(masks) > idx else None
                    
                    x1 = round(float(box[0]))
                    y1 = round(float(box[1]))
                    x2 = round(float(box[2]))
                    y2 = round(float(box[3]))
                    box_xyxy = [x1, y1, x2, y2]
                    
                    segmentation = None
                    if mask is not None:
                        segmentation = self._encode_mask(mask)
                        if segmentation:
                            segmentation = {
                                "size": segmentation["size"],
                                "counts": segmentation["counts"]
                            }
                    
                    if len(json_data["properties"]["JawBones"]) > 0:
                        jawbone = json_data["properties"]["JawBones"][0]
                        if "conditions" not in jawbone:
                            jawbone["conditions"] = {}
                        
                        if "Bone loss" in jawbone["conditions"]:
                            existing_condition = jawbone["conditions"]["Bone loss"]
                            if isinstance(existing_condition["bbox"], list):
                                if isinstance(existing_condition["bbox"][0], list):
                                    existing_condition["bbox"].append(box_xyxy)
                                    existing_condition["segmentation"].append(segmentation)
                                    if isinstance(existing_condition["score"], list):
                                        existing_condition["score"].append(round(score, 2))
                                    else:
                                        existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                else:
                                    existing_condition["bbox"] = [existing_condition["bbox"], box_xyxy]
                                    if "segmentation" in existing_condition:
                                        existing_condition["segmentation"] = [existing_condition["segmentation"], segmentation]
                                    else:
                                        existing_condition["segmentation"] = [None, segmentation]
                                    if isinstance(existing_condition["score"], (int, float)):
                                        existing_condition["score"] = [existing_condition["score"], round(score, 2)]
                                    else:
                                        existing_condition["score"].append(round(score, 2))
                            else:
                                jawbone["conditions"]["Bone loss"] = {
                                    "bbox": [box_xyxy],
                                    "segmentation": [segmentation],
                                    "score": [round(score, 2)]
                                }
                        else:
                            jawbone["conditions"]["Bone loss"] = {
                                "bbox": [box_xyxy],
                                "segmentation": [segmentation],
                                "score": [round(score, 2)]
                            }
                    else:
                        jaw_bone_entry = {
                            "conditions": {
                                "Bone loss": {
                                    "bbox": [box_xyxy],
                                    "segmentation": [segmentation],
                                    "score": [round(score, 2)]
                                }
                            }
                        }
                        json_data["properties"]["JawBones"].append(jaw_bone_entry)
        return json_data

    def detect_mandibular(self, image_path, json_data):
        predictor = self.maskdino_predictors.get("mandibular")
        if not predictor:
            return json_data
            
        img = cv2.imread(image_path)
        predictions, _ = predictor.run_on_image(img)
        
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
                scores = instances.scores.numpy() if instances.has("scores") else []
                masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
                classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
                
                high_conf_indices = np.where(scores > SCORE)[0]
                if "JawBones" not in json_data["properties"]:
                    json_data["properties"]["JawBones"] = []
                
                class_names = {
                    0: "Mandibular canal",
                    1: "Maxillary sinus"
                }
                
                for idx in high_conf_indices:
                    box = boxes[idx]
                    score = float(scores[idx])
                    mask = masks[idx] if len(masks) > idx else None
                    class_id = int(classes[idx])
                    class_name = class_names.get(class_id, "Unknown")
                    
                    x1 = round(float(box[0]))
                    y1 = round(float(box[1]))
                    x2 = round(float(box[2]))
                    y2 = round(float(box[3]))
                    box_xyxy = [x1, y1, x2, y2]
                    
                    segmentation = None
                    if mask is not None:
                        segmentation = self._encode_mask(mask)
                        if segmentation:
                            segmentation = {
                                "size": segmentation["size"],
                                "counts": segmentation["counts"]
                            }
                    
                    if len(json_data["properties"]["JawBones"]) > 0:
                        jawbone = json_data["properties"]["JawBones"][0]
                        if "conditions" not in jawbone:
                            jawbone["conditions"] = {}
                        
                        if class_name in jawbone["conditions"]:
                            existing_condition = jawbone["conditions"][class_name]
                            if isinstance(existing_condition["bbox"], list):
                                if isinstance(existing_condition["bbox"][0], list):
                                    existing_bbox_list = existing_condition["bbox"]
                                    existing_segmentation_list = existing_condition["segmentation"]
                                    if isinstance(existing_condition["score"], list):
                                        existing_score_list = existing_condition["score"]
                                    else:
                                        existing_score_list = [existing_condition["score"]] * len(existing_bbox_list)
                                        existing_condition["score"] = existing_score_list
                                else:
                                    existing_bbox_list = [existing_condition["bbox"]]
                                    if "segmentation" in existing_condition:
                                        existing_segmentation_list = [existing_condition["segmentation"]]
                                    else:
                                        existing_segmentation_list = [None]
                                    if isinstance(existing_condition["score"], (int, float)):
                                        existing_score_list = [existing_condition["score"]]
                                    else:
                                        existing_score_list = existing_condition["score"]
                                    existing_condition["bbox"] = existing_bbox_list
                                    existing_condition["segmentation"] = existing_segmentation_list
                                    existing_condition["score"] = existing_score_list
                            else:
                                jawbone["conditions"][class_name] = {
                                    "bbox": [box_xyxy],
                                    "segmentation": [segmentation],
                                    "score": [round(score, 2)]
                                }
                                continue
                            
                            has_overlap = False
                            for i, existing_box in enumerate(existing_bbox_list):
                                iou = self._calculate_iou_xyxy(existing_box, box_xyxy)
                                if iou > 0:
                                    has_overlap = True
                                    if score > existing_score_list[i]:
                                        existing_bbox_list[i] = box_xyxy
                                        existing_segmentation_list[i] = segmentation
                                        existing_score_list[i] = round(score, 2)
                                    break
                            
                            if not has_overlap:
                                existing_bbox_list.append(box_xyxy)
                                existing_segmentation_list.append(segmentation)
                                existing_score_list.append(round(score, 2))
                        else:
                            jawbone["conditions"][class_name] = {
                                "bbox": [box_xyxy],
                                "segmentation": [segmentation],
                                "score": [round(score, 2)]
                            }
                    else:
                        jaw_bone_entry = {
                            "conditions": {
                                class_name: {
                                    "bbox": [box_xyxy],
                                    "segmentation": [segmentation],
                                    "score": [round(score, 2)]
                                }
                            }
                        }
                        json_data["properties"]["JawBones"].append(jaw_bone_entry)
        return json_data

    def _calculate_iou_xyxy(self, box1, box2):
        if len(box1) == 4 and len(box2) == 4:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
            if intersection_area == 0:
                return 0
            
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            union_area = box1_area + box2_area - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0
            return iou
        else:
            box1_xyxy = self._convert_bbox_to_xyxy(box1)
            box2_xyxy = self._convert_bbox_to_xyxy(box2)
            return self._calculate_iou_xyxy(box1_xyxy, box2_xyxy)

    def _encode_mask(self, mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        try:
            rle = mask_util.encode(np.asfortranarray(mask_uint8))
            
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            
            return rle
        except Exception as e:
            print(f"Error encoding mask: {e}")
            return None

    def _convert_bbox_to_xyxy(self, box_orig):
        if len(box_orig) != 4:
            return box_orig
        
        if isinstance(box_orig[0], (int, float)) and isinstance(box_orig[2], (int, float)):
            if box_orig[0] < box_orig[2] and box_orig[1] < box_orig[3]:
                return [round(float(box_orig[0])), round(float(box_orig[1])), 
                        round(float(box_orig[2])), round(float(box_orig[3]))]
        
        try:
            cx, cy, w, h = box_orig
            x1 = round(float(cx - w / 2))
            y1 = round(float(cy - h / 2))
            x2 = round(float(cx + w / 2))
            y2 = round(float(cy + h / 2))
            return [x1, y1, x2, y2]
        except:
            return box_orig


def process_image(image_path, expert, output_dir, models_config):
    file_name = os.path.basename(image_path)
    image_id = str(os.path.splitext(file_name)[0])
    output_json_path = os.path.join(output_dir, f"{image_id}.json")

    with open(output_json_path, 'r') as f:
        existing_data = json.load(f)
    json_data = existing_data

    # 使用牙齿id视觉专家
    if "teeth_id" in models_config:
        teeth_annotations = expert.detect_teeth_id(image_path)
        json_data["properties"]["Teeth"] = teeth_annotations

    # 使用象限视觉专家
    if "quadrants" in models_config:
        quadrants_annotations = expert.detect_quadrants_with_maskdino(image_path)
        json_data["properties"]["Quadrants"] = quadrants_annotations

    # 使用疾病视觉专家
    if "4diseases" in models_config:
        json_data = expert.detect_4diseases(image_path, json_data)

    # 使用龋齿和充填视觉专家
    if "caries_filling" in models_config:
        json_data = expert.detect_caries_filling(image_path, json_data)

    if "12diseases" in models_config:
        json_data = expert.detect_12diseases(image_path, json_data)

    # 使用MaskDINO骨质流失检测
    if "bone_loss" in models_config:
        json_data = expert.detect_bone_loss(image_path, json_data)

    # 使用MaskDINO 11疾病检测
    if "11diseases" in models_config:
        json_data = expert.detect_11diseases(image_path, json_data)

    if "3periapical_lesions" in models_config:
        json_data = expert.detect_3periapical_lesions(image_path, json_data)

    if "mandibular" in models_config:
        json_data = expert.detect_mandibular(image_path, json_data)

    json_data = convert_numpy_types(json_data)

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    return file_name

def main():
    parser = argparse.ArgumentParser(description="Process dental X-ray images")
    parser.add_argument("--splits", type=int, default=1, help="Number of splits to divide the dataset")
    parser.add_argument("--split_index", type=int, default=1, help="Index of the split to process (1-based)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for processing")
    args = parser.parse_args()
    args.split_index = args.split_index - 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir = Path("/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-images")
    output_dir = Path("/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons")
    output_dir.mkdir(exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))+list(image_dir.glob("*.png"))+list(image_dir.glob("*.JPG"))+list(image_dir.glob("*.PNG"))
    total_files = len(image_files)

    sorted_files = sorted(image_files, key=lambda x: int(x.stem))

    # 根据分割参数分割文件
    if args.split_index >= args.splits:
        raise ValueError(f"Split index {args.split_index} must be less than the number of splits {args.splits}")

    # 计算每个分割的大小并选择相应的文件
    chunk_size = len(sorted_files) // args.splits
    start_idx = args.split_index * chunk_size
    end_idx = (args.split_index + 1) * chunk_size if args.split_index < args.splits - 1 else len(sorted_files)
    selected_files = sorted_files[start_idx:end_idx]

    total_files = len(selected_files)
    print(f"Found {total_files} images to process (split {args.split_index+1}/{args.splits})")

    experts = {
        "teeth_id": "",
        "quadrants": "",
        "caries_filling": "",
        "11diseases": "",
        "12diseases": "",
        "bone_loss": "",
        "3periapical_lesions": "",
        "mandibular": "",
        "4diseases": "",
    }

    for models_config, _ in experts.items():
        print(f"------>model:{models_config}")
        expert = DentalVisualExpert(models_config)

        # 初始化专家模型
        print(f"--------> Initializing dental visual expert: {models_config}")

        # 创建进度条
        with tqdm(total=total_files, desc="Processing images") as pbar:
            # 处理每张图像
            for image_path in selected_files:
                try:
                    file_name = process_image(image_path, expert, output_dir, models_config)
                    pbar.update(1)
                    pbar.set_postfix({"Current": file_name})
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    traceback.print_exc()

        print(f"--------> {models_config} processing complete!")

if __name__ == "__main__":
    main()
