import os
import glob
import json
import pandas as pd
import base64
import random
import re


# 指定 JSON 文件所在的目录
json_dir = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_100/MM-Oral-OPG-vqa-loc-med'  # 替换为你的 JSON 文件目录
images_dir = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_100/MM-Oral-OPG-images'
output_tsv = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/output/test_100/MM-Oral-VQA-Closed-Ended.tsv'




def transform_kv(options_str, answer_str):
    """
    将输入的 Options 和 Answer 转换为指定的字典格式。

    输入:
        options_str: str, 例如 "A) 30 B) 31 C) 32 D) 33"
        answer_str: str, 例如 "B) 31"
    
    输出:
        dict, 转换后的字典格式
    """
    if isinstance(options_str, dict):
        return
    # 使用正则表达式找到所有选项及其内容
    pattern = r"([A-D])\) (.*?)(?= [A-D]\)|$)"
    matches = re.findall(pattern, options_str)

    # 构造 Options 字典
    options_dict = {key: value.strip() for key, value in matches}

    # 解析 Answer 字段
    answer_key = answer_str.split(')')[0]  # 提取答案的选项 (A, B, C, D)

    # 返回结果
    return {
        "Options": options_dict,
        "Answer": answer_key
    }

def encode_image_file_to_base64(image_path):
    """将图片文件编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# 存储所有数据项的列表
all_items = []
index_counter = 0

# 读取所有 JSON 文件
json_files = glob.glob(os.path.join(json_dir, '*.json'))
json_files.sort()
# json_files = json_files[:10]  # 仅处理前10个文件
print(f"找到 {len(json_files)} 个 JSON 文件")

for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 获取图片路径
        file_name = data.get('file_name')
        if not file_name:
            print(f"警告: JSON文件缺少file_name字段: {json_file}")
            continue
            
        image_path = os.path.join(images_dir, file_name)
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在: {image_path}")
            continue
            
        # 将图片编码为 base64
        try:
            image_base64 = encode_image_file_to_base64(image_path)
        except Exception as e:
            print(f"编码图片时出错 {image_path}: {e}")
            continue
        
        # 检查数据结构
        if 'vqa_data' not in data:
            print(f"警告: JSON文件缺少vqa_data字段: {json_file}")
            continue
        
        
        # 处理vqa_data中的所有问题类型和问题
        # for question_type, questions in data['vqa_data'].items():
        #     if question_type in ['loc_open_ended', 'med_open_ended', 'report_generation']:
        #         continue

        questions = data['vqa_data']['med_closed_ended']
        questions_loc = data['vqa_data']['loc_closed_ended']
        if len(questions) < 5:
            questions += random.sample(questions_loc, 5-len(questions))
        else:
            questions = random.sample(questions, 5)

        question_type = 'med_closed_ended'
        # if not isinstance(questions, list):
        #     print(f"警告: 问题类型 {question_type} 不是列表格式: {json_file}")
        #     continue
            
        for q_index, q_data in enumerate(questions):
            try:
                if isinstance(q_data['Options'], str):
                    option_answer_dict = transform_kv(q_data['Options'], q_data['Answer'])
                    q_data['Options'] = option_answer_dict['Options']
                    q_data['Answer'] = option_answer_dict['Answer']

                # 检查必需字段
                if 'Question' not in q_data:
                    print(f"警告: 缺少问题文本: {json_file}, 问题类型: {question_type}, 问题索引: {q_index}")
                    continue
                    
                if 'Options' not in q_data or not isinstance(q_data['Options'], dict):
                    print(f"警告: 缺少选项或选项格式不正确: {json_file}, 问题类型: {question_type}, 问题索引: {q_index}")
                    continue
                
                # 基本信息
                item = {
                    'index': index_counter,
                    'image_id': data.get('image_id', ''),
                    'file_name': file_name,
                    'question': q_data['Question'],
                    'category': q_data.get('Category', question_type),  # 如果没有category，使用question_type
                    'image': image_base64,
                    'type': '选择'  # 单项选择题
                }

                # 处理选项 (ABCD)
                options = q_data['Options']
                option_keys = []
                
                for key, value in options.items():
                    if key in ['A', 'B', 'C', 'D']:
                        option_number = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}[key]
                        item[f'option{option_number}'] = value
                        option_keys.append(key)
                
                # 检查是否有四个选项
                if len(option_keys) != 4:
                    print(f"警告: 选项数量不是4个: {json_file}, 问题类型: {question_type}, 问题索引: {q_index}")
                    # 填充缺失的选项
                    for opt in ['A', 'B', 'C', 'D']:
                        if opt not in option_keys:
                            option_number = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}[opt]
                            item[f'option{option_number}'] = f"[缺失选项 {opt}]"
                
                # 添加答案
                if 'Answer' in q_data and q_data['Answer'] in ['A', 'B', 'C', 'D']:
                    item['answer'] = q_data['Answer']
                else:
                    print(f"警告: 缺少有效答案或答案格式不正确: {json_file}, 问题类型: {question_type}, 问题索引: {q_index}")
                    item['answer'] = ''  # 测试集可能没有答案
                
                # 添加到列表
                all_items.append(item)
                index_counter += 1
                
                # 每处理100个问题打印一次进度
                if index_counter % 100 == 0:
                    print(f"已处理 {index_counter} 个问题...")
            except Exception as e:
                print(f"处理问题时出错: {json_file}, 问题类型: {question_type}, 问题索引: {q_index}, 错误: {e}")

    except Exception as e:
        print(f"处理JSON文件时出错: {json_file}, 错误: {e}")

# 如果没有收集到数据，报错
if not all_items:
    raise ValueError("没有收集到任何有效数据！请检查JSON文件格式和路径。")

# 创建 DataFrame 并保存为 TSV
df = pd.DataFrame(all_items)

# 确保必需字段在 DataFrame 中
required_fields = ['index', 'image', 'question', 'answer', 'option1', 'option2', 'option3', 'option4']
for field in required_fields:
    if field not in df.columns:
        if field != 'answer':  # answer 可以在测试集中缺失
            print(f"警告: 缺少必需字段 {field}，将添加空值")
            df[field] = ''

# 保存为 TSV 文件
df.to_csv(output_tsv, sep='\t', index=False)

print(f"处理完成！已将数据转换为 TSV 格式并保存到: {output_tsv}")
print(f"总共处理了 {index_counter} 个问题（来自 {len(json_files)} 个JSON文件）")