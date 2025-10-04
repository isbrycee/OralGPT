import json

def modify_file_names(
    input_file_path: str,       # 输入JSON文件路径（原始文件）
    output_file_path: str,      # 输出JSON文件路径（修改后保存的文件）
    old_substring: str,         # 要替换的旧子字符串（比如"images/"）
    new_substring: str          # 替换后的新子字符串（比如"processed_images/"）
) -> None:
    """
    批量修改JSON数据中所有条目的file_name字段（替换指定子字符串）
    
    示例：将file_name从"images/xxx.jpg"改为"new_images/xxx.jpg"
    """
    # 1. 读取原始JSON数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)  # 加载为Python列表（每个元素是字典）
    
    # 2. 遍历所有条目，修改file_name字段
    modified_count = 0
    for item in original_data:
        if 'file_name' in item:  # 确保条目包含file_name字段
            original_name = item['file_name']
            # 替换子字符串（比如把"images/"换成"new_images/"）
            new_name = original_name.replace(old_substring, new_substring)
            item['file_name'] = new_name
            
            # 统计修改次数（可选）
            if new_name != original_name:
                modified_count += 1
    
    # 3. 保存修改后的数据到新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)  # 保留格式和中文
    
    # 打印结果反馈
    print(f"✅ 处理完成！共修改 {len(original_data)} 条数据，其中 {modified_count} 条的file_name被更新。")
    print(f"💾 修改后的文件已保存至：{output_file_path}")


if __name__ == "__main__":
    mode = 'train'
    modify_file_names(
        input_file_path=f"/home/jinghao/projects/x-ray-VLM/RGB/intraoral_image_for_location_counting/intraoral_image_for_location_counting_{mode}_question.json",    # 原始文件路径
        output_file_path=f"/home/jinghao/projects/x-ray-VLM/RGB/intraoral_image_for_location_counting/intraoral_image_for_location_counting_{mode}_question_new.json",   # 修改后保存的文件路径
        old_substring="images/",                 # 要替换的旧路径（比如"images/"）
        new_substring="MMOral-Omni/2.1_intraoral_image_location/"              # 替换后的新路径（比如"new_images/"）
    )