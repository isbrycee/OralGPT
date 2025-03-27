import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import argparse
import json
from PIL import Image
import sys

# Add MaskDINO module to path
sys.path.append("/hpc2hdd/home/yfan546/workplace/xray_teeth/MaskDINO") # Replace with your MaskDINO directory path

# Import MaskDINO modules
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from demo.predictor import VisualizationDemo

def load_maskdino_model(config_file, model_weights, device='cuda'):
    """Load MaskDINO model"""
    # Configure MaskDINO
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    
    # Create predictor
    predictor = VisualizationDemo(cfg)
    
    return cfg, predictor

def load_category_names(coco_path):
    """Load category names from COCO format"""
    
    class_names = {
        5: "Mandibular canal",
        11: "Maxillary sinus"
        }
    
    if coco_path:
        with open(coco_path) as f:
            data = json.load(f)
            if 'categories' in data:
                id2name = {cat['id']: cat['name'] for cat in data['categories']}
            else:
                # Try direct use, but skip non-numeric keys
                id2name = {}
                for k, v in data.items():
                    try:
                        id2name[int(k)] = v
                    except ValueError:
                        continue
    elif class_names is not None:
        return {int(k): v for k, v in class_names.items()}
    
    return id2name

def run_inference(predictor, image_path):
    """Run MaskDINO inference"""
    # Read image
    img = cv2.imread(image_path)
    
    # Run inference
    predictions, visualized_output = predictor.run_on_image(img)
    
    return predictions, img

def visualize_predictions(image, predictions, id2name, output_path, confidence_threshold=0.3):
    """Visualize detection results"""
    # Convert image from BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(image_rgb)
    ax.axis('off')
    ax.set_title(f"MaskDINO Detection Results: {os.path.basename(output_path)}")
    
    # Get detections
    if "instances" in predictions:
        instances = predictions["instances"].to("cpu")
        
        # Get predictions above threshold
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
            scores = instances.scores.numpy() if instances.has("scores") else []
            classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
            masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
            
            # Filter by confidence threshold
            high_conf_indices = np.where(scores > confidence_threshold)[0]
            
            print(f"Detected {len(high_conf_indices)} objects with confidence threshold {confidence_threshold}")
            
            # Use different colors for different classes
            colors = plt.cm.rainbow(np.linspace(0, 1, max(len(id2name), 1)))
            
            # Plot each detection
            for i in high_conf_indices:
                # Get box (xyxy format)
                box = boxes[i]
                score = scores[i]
                class_id = int(classes[i]) + 1  # Adjust for 1-indexed classes
                
                # Get class name
                label_name = id2name.get(class_id, f"Unknown ({class_id})")
                
                # Select color
                color = colors[class_id % len(colors)]
                
                # Draw bounding box
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Draw mask if available
                if i < len(masks):
                    mask = masks[i]
                    # Add mask as transparent overlay
                    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
                    mask_rgba[:, :, 0] = color[0]
                    mask_rgba[:, :, 1] = color[1]
                    mask_rgba[:, :, 2] = color[2]
                    mask_rgba[:, :, 3] = mask * 0.5  # 50% transparency
                    ax.imshow(mask_rgba, alpha=0.5)
                
                # Add text label
                text = f"{label_name} ({score:.2f})"
                text_x = x1
                text_y = y1 - 5
                
                plt.text(
                    text_x, text_y, text,
                    bbox=dict(facecolor=color, alpha=0.5),
                    fontsize=9, color='white'
                )
                
                # Print detection info
                print(f"Object {i+1}: {label_name}, confidence: {score:.3f}, box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Save result
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MaskDINO Single Image Inference and Visualization')
    parser.add_argument('--image', type=str, default="./000043.jpg", help='Input image path')
    parser.add_argument('--config', type=str, default="/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/config/DINO/maskdino_Swinl_bs16_50ep_4s_dowsample1_2048_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.yaml", help='Model config file path')
    parser.add_argument('--checkpoint', type=str, default="/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/model_weights/Teeth_Visual_Experts_Maskdino_Swinl_panoramic_x-ray_Mandibular_Canal_Maxillary_Sinus.pth.1", help='Model weights path')
    parser.add_argument('--coco_names', type=str, default="/hpc2hdd/home/yfan546/workplace/xray_teeth/DINO/teeth_data/teeth_x-ray_11diseases_numImages8423_ins_seg_coco/annotations_coco.json", help='COCO category file path')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default="./outputs", help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Inference device')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading model: {args.checkpoint}")
    # Load MaskDINO model
    _, predictor = load_maskdino_model(args.config, args.checkpoint, args.device)
    
    # Load category names
    id2name = load_category_names(args.coco_names)
    print(f"Loaded {len(id2name)} category names")
    
    # Process image
    image_path = args.image
    print(f"Processing image: {image_path}")
    
    # Run inference
    predictions, original_image = run_inference(predictor, image_path)
    
    # Set output path
    output_filename = f"maskdino_{os.path.basename(image_path).split('.')[0]}.png"
    output_path = os.path.join(args.output, output_filename)
    
    # Visualize results
    visualize_predictions(original_image, predictions, id2name, output_path, args.confidence)

if __name__ == "__main__":
    main()
