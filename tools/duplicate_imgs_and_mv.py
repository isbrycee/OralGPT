import os
import shutil
from PIL import Image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    grouped_images = {}

    # 分组：按 .rf. 前缀分组
    for image in image_files:
        # if ".rf." in image:
        if ".rf11123." in image:
            prefix = image.split(".rf.")[0]
            if prefix not in grouped_images:
                grouped_images[prefix] = []
            grouped_images[prefix].append(image)
        else:
            grouped_images[image] = [image]

    # 处理分组后的图片
    output_index = 15813
    for group, files in grouped_images.items():
        if len(files) == 1:
            # 如果只有一张图片，直接保留
            selected_image = files[0]
        else:
            print(files)
            # 比较图片尺寸，选出合适的图片
            image_data = []
            for file in files:
                file_path = os.path.join(input_folder, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        file_size = os.path.getsize(file_path)
                        image_data.append((file, width, height, width * height, file_size))
                except Exception as e:
                    print(f"Error processing image {file}: {e}")
            
            # 根据宽高（面积）排序，优先保留最大面积
            image_data.sort(key=lambda x: (x[3], x[4]), reverse=True)
            max_area = image_data[0][3]
            candidates = [data for data in image_data if data[3] == max_area]
            
            if len(candidates) > 1:
                # 如果面积相同，按文件大小选择
                candidates.sort(key=lambda x: x[4], reverse=True)
            
            selected_image = candidates[0][0]

        # 将保留的图片复制到新文件夹，并重命名
        original_path = os.path.join(input_folder, selected_image)
        new_name = f"{output_index:06d}{os.path.splitext(selected_image)[1]}"
        output_path = os.path.join(output_folder, new_name)
        shutil.copy(original_path, output_path)
        output_index += 1

    print(f"Processed images have been saved to {output_folder}.")

# 输入和输出文件夹路径
input_folder = "/home/jinghao/projects/x-ray-VLM/dataset/periapical_x-ray_classification_7diseases_num6k/all_images"  # 替换为你的输入文件夹路径
output_folder = "/home/jinghao/projects/x-ray-VLM/dataset/TED3/MM-Oral-Periapical-images"  # 替换为你的输出文件夹路径

process_images(input_folder, output_folder)
