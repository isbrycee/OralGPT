import os
import json
import jieba
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def safe_convert(input_path):
    img = Image.open(input_path)
    
    # 处理调色板图像
    if img.mode == 'P':
        # 创建临时全白背景
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
        img = background
    
    # 处理透明通道
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    
    # 转换为灰度
    gray_img = img.convert('L')
    return gray_img

def create_tooth_mask(input_path, output_path="tooth_mask.png"):
    """
    创建牙齿形状的二值化mask
    参数：
    input_path: 输入RGB图像路径
    output_path: 输出mask保存路径
    """
    # 读取并转换图像
    img = safe_convert(input_path)
    
    img_array = np.array(img)
    # 自动阈值二值化（处理黑白反转）
    _, binary = cv2.threshold(img_array, 0, 255, 
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 创建一个空白图像用于填充
    filled_image = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 填充轮廓内部区域
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    # 将填充后的区域与原图结合
    result = cv2.bitwise_or(binary, filled_image)

    # 保存结果
    cv2.imwrite('binary_output.png', result)

    # 生成最终二值mask
    final_mask = (result == 255).astype(np.uint8)  # 1: 牙齿区域，0: 背景

    # 保存结果
    Image.fromarray(final_mask * 255).save(output_path)
    return final_mask

def generate_tooth_wordcloud(json_folder, mask_path, output_path="tooth_wordcloud.png"):
    """
    生成牙齿形状词云
    
    参数：
    json_folder: JSON文件夹路径
    mask_path: 牙齿形状蒙版图片路径
    output_path: 输出图片路径
    """
    # 读取所有JSON文件
    all_text = ""
    
    # 1. for medical report
    # for filename in os.listdir(json_folder):
    #     if filename.endswith(".json"):
    #         with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
    #             data = json.load(f)
    #             all_text += data.get("med_report", "") + " "

    # 2. for single vqa
    # for filename in os.listdir(json_folder):
    #     if filename.endswith(".json"):
    #         with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
    #             data = json.load(f)
    #             vqa_data = data.get("vqa_data", "")
    #             for item in vqa_data['loc_closed_ended']:
    #                 all_text += item['Question'] + " "
    #             for item in vqa_data['loc_open_ended']:
    #                 all_text += item['Question'] + " "
    #             for item in vqa_data['med_closed_ended']:
    #                 all_text += item['Question'] + " "
    #             for item in vqa_data['med_open_ended']:
    #                 all_text += item['Question'] + " "
    #                 all_text += item['Answer'] + " "

    # 3. for multi conversations
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                multi_conv = data.get("conversations", "")
                for content in multi_conv:
                    
                    all_text += content['content'] + " "


    # 中文分词处理
    seg_text = " ".join(jieba.cut(all_text))

    # 加载牙齿形状蒙版
    mask = np.array(Image.open(mask_path))
    mask = 255 - mask

    # 生成词云
    wc = WordCloud(
        font_path='MSYH.TTC',  # 使用微软雅黑字体
        background_color="white",
        mask=mask,
        max_words=350,
        max_font_size=100
    )
    wc.generate(seg_text)

    # 保存结果
    wc.to_file(output_path)
    print(f"词云已生成至：{output_path}")

# 使用示例
if __name__ == "__main__":
    # 需要用户自己准备的参数
    json_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/FINAL/train/MM-Oral-OPG-multi-turn-conv"  # JSON文件夹路径
    mask_path = "tooth_mask3.jpg"     # 牙齿形状蒙版图片路径
    mask_path_bi = "tooth_mask_binary.png"
    create_tooth_mask(mask_path, mask_path_bi)
    generate_tooth_wordcloud(json_folder, mask_path_bi)