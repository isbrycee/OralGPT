import base64
import os
import pandas as pd
from PIL import Image
from io import BytesIO

def save_images_from_tsv(filepath, output_folder):
    # 读取 tsv 文件
    df = pd.read_csv(filepath, sep='\t')
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 处理每一行
    for idx, row in df.iterrows():
        index_str = str(row['index'])
        image_data = row['image']
        
        # 判断 image_data 是单张图片还是列表
        try:
            # 尝试解析为列表（假设以字符串形式存的列表）
            image_list = eval(image_data)
            if not isinstance(image_list, list):
                image_list = [image_data]
        except:
            # 解析异常说明是单个 base64 字符串
            image_list = [image_data]
        
        # 逐张保存图片
        for i, img_str in enumerate(image_list):
            try:
                img_bytes = base64.b64decode(img_str)
                image = Image.open(BytesIO(img_bytes))
                
                # 获取图片格式作为后缀
                ext = image.format.lower() if image.format else 'png'
                
                # 多图加序号，否则直接用 index
                if len(image_list) > 1:
                    filename = f"{index_str}_{i}.{ext}"
                else:
                    filename = f"{index_str}.{ext}"
                
                save_path = os.path.join(output_folder, filename)
                image.save(save_path)
                print(f"Saved image to: {save_path}")
            except Exception as e:
                print(f"Failed to save image at index {index_str} image {i}: {e}")

# 示例调用
save_images_from_tsv('/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new_II_loc_cepha_intraoral_image-level_diagnosis_PA_Histo_Video_RegionLevelDiagnosis_valid_cleaned_finalize_category_cleaned_woTE_resizeFDTooth_resizeGingivitis_resizePINormality.tsv',
                      '/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral-OMNI-Bench-Images')

