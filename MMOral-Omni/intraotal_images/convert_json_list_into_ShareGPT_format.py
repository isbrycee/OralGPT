import json

def convert_json_format(input_file_path, output_file_path):
    """
    将原始JSON格式转换为目标的对话+图像格式
    
    参数:
        input_file_path: 输入JSON文件路径（原始数据）
        output_file_path: 输出JSON文件路径（目标格式）
    """
    # 1. 读取原始JSON数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)  # 原始数据是list，每个元素是dict
    
    # 2. 初始化目标数据列表
    target_data = []
    
    # 3. 遍历原始数据的每个元素，构造目标结构
    for item in original_data:
        # 提取所需字段（确保原始数据包含这些键，否则会报错）
        question = item.get("question", "")  # 若缺失则默认空字符串
        caption = item.get("caption", "")
        file_name = item.get("file_name", "")
        file_name = file_name.replace('images/', 'MMOral-Omni/2.1_intraoral_image_location/')
        
        # 构造目标条目
        target_entry = {
            "conversations": [
                {"from": "human", "value": '<image> '+question},  # human的问题
                {"from": "gpt", "value": caption}      # gpt的回答（这里用caption填充）
            ],
            "images": [file_name]  # 图像路径列表（原始file_name）
        }
        
        # 添加到目标数据
        target_data.append(target_entry)
    
    # 4. 写入目标JSON文件（保留中文、美化格式）
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)


# ------------------- 示例调用 -------------------
if __name__ == "__main__":
    # 替换为你的输入/输出文件路径
    INPUT_JSON_PATH = "/home/jinghao/projects/x-ray-VLM/RGB/intraoral_image_for_location_counting/intraoral_image_for_location_counting_train_question.json"  # 原始数据文件
    OUTPUT_JSON_PATH = "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/2.1_sft_intraoralImage_location_2datasets_shareGPT.json"  # 转换后的文件
    
    convert_json_format(INPUT_JSON_PATH, OUTPUT_JSON_PATH)
    print(f"转换完成！结果已保存至：{OUTPUT_JSON_PATH}")
