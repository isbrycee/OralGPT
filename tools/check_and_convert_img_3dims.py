import os
import shutil
from PIL import Image
from pathlib import Path

def convert_single_channel_images(input_dir, output_dir):
    """
    处理图像通道转换的脚本
    :param input_dir: 输入图片文件夹路径
    :param output_dir: 输出图片文件夹路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    # 遍历输入目录
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 跳过非文件项和无效扩展名
        if not os.path.isfile(input_path):
            continue
        if Path(filename).suffix.lower() not in valid_extensions:
            print(f"跳过非图片文件: {filename}")
            continue
            
        try:
            # 直接复制非图片文件（如果前面检查不严格）
            if Path(filename).suffix.lower() not in valid_extensions:
                shutil.copy(input_path, output_path)
                continue

            with Image.open(input_path) as img:
                # 检查通道数
                if img.mode in ('L', 'LA'):
                    # 处理单通道图像
                    if img.mode == 'L':
                        converted = img.convert('RGB')
                    elif img.mode == 'LA':
                        converted = Image.merge('RGBA', [img.split()[0]]*3 + [img.split()[1]])
                    
                    # 保存转换后的图像
                    converted.save(output_path, quality=95, **img.info)
                    print(f"已转换: {filename} ({img.mode} -> {converted.mode})")
                else:
                    # 直接复制多通道图像（使用shutil保留原文件）
                    shutil.copy(input_path, output_path)
                    # print(f"直接复制: {filename} ({img.mode} 通道)")
                    
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='图像通道转换工具')
    parser.add_argument('--input', required=True, help='输入图片目录')
    parser.add_argument('--output', required=True, help='输出图片目录')
    
    args = parser.parse_args()
    
    convert_single_channel_images(args.input, args.output)