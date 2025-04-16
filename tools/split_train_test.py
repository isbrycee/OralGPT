import os
import shutil

def split_files(folder_a, folder_b, output_dir):
    # 创建输出的 train 和 test 文件夹
    train_dir = os.path.join(output_dir, "train", os.path.basename(folder_a))
    test_dir = os.path.join(output_dir, "test", os.path.basename(folder_a))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取文件夹 B 中的文件名前缀
    b_prefixes = set(os.path.splitext(f)[0] for f in os.listdir(folder_b) if os.path.isfile(os.path.join(folder_b, f)))

    # 遍历文件夹 A，将文件根据条件分到 train 和 test 中
    for file_name in os.listdir(folder_a):
        file_path = os.path.join(folder_a, file_name)
        if not os.path.isfile(file_path):
            continue

        # 获取文件的前缀（不含扩展名）
        file_prefix = os.path.splitext(file_name)[0]

        # 判断该文件是属于 test 还是 train
        if file_prefix in b_prefixes:
            dest_dir = test_dir
        else:
            dest_dir = train_dir

        # 复制文件到目标文件夹
        shutil.move(file_path, os.path.join(dest_dir, file_name))

    print(f"文件已成功拆分到 {train_dir} 和 {test_dir}")

# 示例用法
folder_a = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_4k_bak/MM-Oral-OPG-images"  # 末尾不能有 / 
folder_b = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_100/MM-Oral-OPG-images"  # 替换为文件夹 B 的路径
output_dir = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/1111111"  # 替换为输出文件夹的路径

split_files(folder_a, folder_b, output_dir)