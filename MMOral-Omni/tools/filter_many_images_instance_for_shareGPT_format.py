import json

# 假设你的 JSON 文件路径是 'data.json'
file_path = '/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/0.1_sft_multimodal_ALL_shareGPT.json'

# 读取 JSON 文件
with open(file_path, 'r') as file:
    data = json.load(file)

# 统计所有 images 字段中 list 的长度
total_images_length = sum(len(item["images"]) for item in data)

# 打印第 100 到 120 个元素的 images 字段长度
lengths_100_to_120 = [len(data[i]["images"]) for i in range(3250, min(3300, len(data)))]

print("总 images 字段长度:", total_images_length)
print("第 100 到 120 个元素的 images 字段长度:", lengths_100_to_120)