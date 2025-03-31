import os
import shutil
from PIL import Image

def move_square_images(source_folder, target_folder):
    """
    将宽高比为 1 的图片从 source_folder 移动到 target_folder
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        # 检查文件是否为图片
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                # 如果宽高比为 1，移动文件
                if width == height:
                    shutil.move(file_path, os.path.join(target_folder, filename))
                    print(f"Moved: {filename}")
        except Exception as e:
            # 如果文件无法打开为图片，跳过
            print(f"Skipping {filename}: {e}")

# 使用示例
source_folder = "path/to/source/folder"  # 替换为你的源文件夹路径
target_folder = "path/to/target/folder"  # 替换为你的目标文件夹路径

move_square_images(source_folder, target_folder)
