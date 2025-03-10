import cv2
import os
import numpy as np

def template_matching(image_path, template_path, threshold=0.65):
    # 读取图片和模板
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    print(template)
    if img is None or template is None:
        return False

    # 模板匹配
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # 检查匹配结果是否高于阈值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val >= threshold

def find_images_by_template(folder_path, template_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            
            # 检测图片是否符合模板
            if template_matching(file_path, template_path):
                print(f"符合条件的图片: {filename}")
                # 复制图片到输出文件夹
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cv2.imread(file_path))

# 设置文件夹路径
input_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-OPG-images" # 替换为你的输入文件夹路径
output_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-OPG-filter-low-quality-images"  # 替换为你的输出文件夹路径
template_image = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-OPG-images/000088.jpg"  # 替换为你的模板图片路径

# 运行脚本
find_images_by_template(input_folder, template_image, output_folder)
