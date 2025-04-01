import json
import os
import re
from ast import literal_eval

Questions_Template = [
    'Which {} are visible in the panoramic dental image?',
    'Which {} can be detected in the panoramic image?,',
    'Identify the {} in the X-ray image.',
    'Detect {} in the panoramic image.',
    'Output the positions of {} in the panoramic X-ray image.',
    'Which {} have been accurately detected in the panoramic image?',
    'Please detect the {} in the panoramic image.',
    'Which areas show the {} in the panoramic image?',
    'Identify areas showing the {} in the panoramic radiograph.',
    'Please accurately identify the {} in the X-ray image.'
]


def parse_medical_string(input_str):
    """解析医疗检测字符串为结构化字典"""
    pattern = r"""
        ([A-Za-z\s]+)\s+    # 匹配分类名称
        (?:\(.*?\):\s*)     # 跳过括号内容
        (\[.*?\])\s*        # 捕获检测条目
        (?=\n\S|\Z)         # 前瞻判断结束位置
    """
    
    result_dict = {}
    
    # 分割主分类
    for match in re.finditer(pattern, input_str, re.VERBOSE | re.DOTALL):
        category, entries_str = match.groups()
        
        # 清理分类名称
        clean_category = re.sub(r'\s+', ' ', category).strip()
        
        # 解析条目
        entries = []
        entry_pattern = r"\{([^}]+)\}"
        for entry_match in re.finditer(entry_pattern, entries_str):
            entry_str = entry_match.group(1)
            
            # 转换键值对
            entry_dict = {}
            kv_pattern = r"'(\w+)':\s*(.+?)(?=,\s*'|$)"
            for kv in re.finditer(kv_pattern, entry_str):
                key, value = kv.groups()
                
                # 值类型转换
                try:
                    parsed_value = literal_eval(value.strip())
                except:
                    parsed_value = value.strip()
                
                if key != 'score':
                    entry_dict[key] = parsed_value
            
            if entry_dict:
                entries.append(entry_dict)
        
        result_dict[clean_category] = entries
    
    return result_dict


def extract_field_from_jsons(input_folder, output_folder):
    """
    从输入文件夹中的所有JSON文件中提取指定字段，并保存到输出文件夹中的一个汇总JSON文件中
    
    :param input_folder: 包含输入JSON文件的文件夹路径
    :param output_folder: 输出文件夹路径
    :param field_name: 要提取的字段名，默认为'xxx'
    """
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 准备汇总数据
    extracted_data = {}
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(input_folder, filename)
            
            try:
                # 读取JSON文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                loc_caption = data['loc_caption'].split('including:\n')[1].strip()
                result = parse_medical_string(loc_caption)
                print(result)
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # # 准备输出文件路径
    # output_filename = f"extracted_{field_name}_summary.json"
    # output_path = os.path.join(output_folder, output_filename)
    
    # # 保存提取的数据到新JSON文件
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    
    # print(f"成功提取并保存数据到 {output_path}")
    # print(f"共处理了 {len(extracted_data)} 个文件中的 '{field_name}' 字段")


# 使用示例
if __name__ == "__main__":
    input_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report'
    output_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report_sft'
    
    extract_field_from_jsons(input_folder, output_folder)