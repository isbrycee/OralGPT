import os
import cv2
from natsort import natsorted

def images_to_video(subfolder_path, frame_interval=50, fps=30):
    # 读取所有 png 文件
    images = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.png')]
    images = natsorted(images)  # 按自然顺序排序

    if not images:
        print(f"跳过：{subfolder_path}（没有 PNG 图片）")
        return

    # 采样间隔取帧
    sampled_images = images[::frame_interval]
    first_img_path = os.path.join(subfolder_path, sampled_images[0])
    frame = cv2.imread(first_img_path)

    if frame is None:
        print(f"跳过：无法读取 {first_img_path}")
        return

    height, width, _ = frame.shape

    # 输出视频路径
    output_path = os.path.join(subfolder_path, "output.mp4")

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, img_name in enumerate(sampled_images):
        img_path = os.path.join(subfolder_path, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"警告：无法读取 {img_path}，跳过")
            continue
        out.write(frame)
        print(f"正在写入帧 {i + 1}/{len(sampled_images)} 到 {output_path}")

    out.release()
    print(f"✅ 视频已保存: {output_path}")


def process_main_folder(main_folder, frame_interval=50, fps=30):
    subfolders = [os.path.join(main_folder, d, 'GT') for d in os.listdir(main_folder)
                  if os.path.isdir(os.path.join(main_folder, d))]

    for subfolder in subfolders:
        print(f"\n🔹 处理子文件夹: {subfolder}")
        images_to_video(subfolder, frame_interval, fps)


if __name__ == "__main__":
    # 你可以修改这里为你的主文件夹路径
    main_folder_path = r"/home/jinghao/projects/x-ray-VLM/RGB/intraoral_video_for_comprehension/Vident-real/test"
    process_main_folder(main_folder_path, frame_interval=1, fps=55)
