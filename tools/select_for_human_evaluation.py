import os
import random
import shutil

def copy_random_files(json_folder, image_folder, new_json_folder, new_image_folder, sample_size=100, seed=None):
    # 如果提供了种子，则固定随机数生成器的种子
    if seed is not None:
        random.seed(seed)

    # 获取 JSON 和图像文件的列表
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 提取文件名前缀
    json_prefixes = set(os.path.splitext(f)[0] for f in json_files)
    image_prefixes = set(os.path.splitext(f)[0] for f in image_files)

    # 找到两者共有的前缀
    common_prefixes = list(json_prefixes & image_prefixes)

    # 如果共有前缀数量少于要抽取的数量，抛出错误
    if len(common_prefixes) < sample_size:
        raise ValueError(f"共有的文件前缀数量不足 {sample_size} 个，仅找到 {len(common_prefixes)} 个。")

    # 随机抽取指定数量的前缀
    selected_prefixes = random.sample(common_prefixes, sample_size)

    # 创建目标文件夹（如果不存在）
    os.makedirs(new_json_folder, exist_ok=True)
    os.makedirs(new_image_folder, exist_ok=True)

    # 复制文件
    for prefix in selected_prefixes:
        json_file = os.path.join(json_folder, f"{prefix}.json")
        image_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_image_file = os.path.join(image_folder, f"{prefix}{ext}")
            if os.path.exists(potential_image_file):
                image_file = potential_image_file
                break
        
        if not os.path.exists(json_file) or image_file is None:
            print(f"跳过文件 {prefix}，因为对应的 JSON 或图片文件不存在。")
            continue

        # 复制 JSON 文件
        shutil.copy(json_file, os.path.join(new_json_folder, f"{prefix}.json"))
        
        # 复制图像文件
        shutil.copy(image_file, os.path.join(new_image_folder, os.path.basename(image_file)))

    print(f"成功抽取 {sample_size} 个文件并复制到新文件夹中！")

# 使用示例
if __name__ == "__main__":
    json_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/MM-Oral-OPG-jsons_latestv3_wloc_wreport"  # 替换为 JSON 文件夹路径
    image_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/MM-Oral-OPG-images"  # 替换为图像文件夹路径
    new_json_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/for_human_evaluation/jsons"  # 替换为新的 JSON 目标文件夹路径
    new_image_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/for_human_evaluation/images"  # 替换为新的图像目标文件夹路径
    
    # 设置随机种子
    seed = 42  # 固定随机种子，确保结果可复现
    copy_random_files(json_folder, image_folder, new_json_folder, new_image_folder, sample_size=100, seed=seed)