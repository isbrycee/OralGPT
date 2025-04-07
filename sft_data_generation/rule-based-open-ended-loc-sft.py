import json
import os
import re
from ast import literal_eval
import random
from random import choice
from collections import defaultdict


Questions_Template = [
    'Which {} is/are visible in the panoramic dental image?',
    'Which {} can be detected in the panoramic image?',
    'Identify the {} in the X-ray image.',
    'Detect {} in the panoramic image.',
    'Output the positions of {} in the panoramic X-ray image.',
    'Which {} could be detected in the panoramic image?',
    'Please detect the {} in the panoramic image.',
    'Which areas show the {} in the panoramic image?',
    'Identify areas showing the {} in the panoramic radiograph.',
    'Please accurately identify the {} in the X-ray image.'
]

key_map = {'Teeth visibility with center points': 'teeth',
           'Wisdom teeth detection': 'wisdom tooth', 
           'wisdom impacted teeth detection': 'non-wisdom tooth but impacted tooth',
           'Missing teeth detection': 'missing teeth',
           'Dental caries detection': 'dental caries',
           'Periapical lesions detection': 'periapical lesions',
           'Historical treatments': 'historical treatments',
           'Bone loss detection': 'bone loss',
           'Mandibular canal visibility': 'mandibular canal',
           'Maxillary sinuses visibility': 'maxillary sinus',
           }

Questions_Time_Acquisition_Template = [
    'When was this dental panoramic X-ray taken?',
    'What is the date and time recorded on the dental X-ray?',
    'Can you provide the timestamp for this panoramic dental image?',
    'What is the imaging date and time shown on this X-ray?'
    'What does the timestamp on the X-ray indicate?'
    'Could you tell me the exact date and time this dental X-ray was captured?'
    'When does the panoramic X-ray show it was taken?'
    'What is the specific timestamp visible on the X-ray image?'
]

Questions_Time_Acquisition_Reject_Template = [
    'No, there is no timestamp on the image.',
    "No, it's not shown on the image.",
    'No, those details are not included.',
    'Sorry, the date and time are not provided.',
    'Sorry, the image does not show this information.',
    "No, it's not visible."
]

one_tooth_all_properties_Template = [
    'What findings can be observed in the panoramic radiograph regarding tooth #{}?',
    'Please provide a full description of the condition of tooth #{}.',
    'What radiographic features are visible in tooth #{} on the panoramic X-ray?',
    'Describe the overall status of tooth #{} based on the panoramic radiograph findings.',
    'What is the overall structural integrity of tooth #{} as seen on the panoramic radiograph?',
    'Does tooth #{} exhibit any signs of pathology or structural compromise on the panoramic radiograph?',
]

one_disease_all_teeth_properties_Template = [
    'Which teeth demonstrate radiographic features associated with the {} on the panoramic radiograph?',
    'What are the characteristic radiographic signs of the {} observed in multiple teeth on the panoramic radiograph?',
    'Does the panoramic radiograph reveal any specific tooth involvement trends related to {}?',
    'Are there any teeth showing advanced progression of the {} compared to others in the panoramic X-ray?',
    'What are the characteristic features of the {} visible in multiple teeth on the panoramic radiograph?',
    'Does the panoramic X-ray show any systemic involvement of the {} across the entire dentition?',
]

all_Pathological_Findings_Template = [
    'What pathological findings are evident on the panoramic radiograph? Provide a comprehensive list.', 
    'Describe all abnormal radiographic features observed across the dentition in this panoramic X-ray.',
    'Which teeth exhibit signs of pathology on the panoramic radiograph? Summarize the key findings.', 
    'What are the most significant radiographic anomalies present in this panoramic radiograph?',  
    'Identify and categorize all detectable pathological conditions in this panoramic radiograph.', 
    'Are there any pathological findings on the panoramic radiograph? List them systematically.',  
]

all_impacted_teeth_Template = [
    'Which teeth are impacted as seen on the panoramic radiograph? Specify their positions.',
    'Describe the radiographic characteristics of impacted teeth visible on this panoramic X-ray.',  
    'How many impacted teeth are present in the entire dentition based on the panoramic radiograph?',  
    'Identify and locate all impacted teeth visible on the panoramic radiograph.',
    'Does the panoramic X-ray show any systemic involvement of the impacted teeth across the entire dentition?',
    'Please identify all impacted teeth visible on this panoramic radiograph.',
]


all_jawbone_Template = [
    'Identify and describe all visible jawbone structures on this panoramic radiograph.',
    'What radiographic features of the maxilla and mandible are evident in this panoramic X-ray?',
    'Describe any signs of alveolar bone loss or resorption observed in the maxilla or mandible.',
    'Does the panoramic radiograph reveal the maxilla and mandible structure? Specify their locations.',
    'Assess the integrity of the jawbone-related structures as seen on this panoramic X-ray.',
    'Does the panoramic X-ray demonstrate any jawbone-related structures?',
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


def extract_one_tooth_all_properties(dict_data):
    # for one_tooth_all_properties_Template
    tooth_data = defaultdict(list)
    return_q_a_list = []
    # 遍历每个类别，将信息按 tooth_id 分类
    for category, items in dict_data.items():
        for item in items:
            tooth_id = item.get('tooth_id')  # 获取 tooth_id
            if tooth_id:
                # 去掉 tooth_id 字段，将其余字段加入到结果中
                item_data = {k: v for k, v in item.items() if k != 'tooth_id'}
                if category == 'Teeth visibility with center points':
                    _category = 'Teeth position'
                    tooth_data[tooth_id].append({_category: item_data})
                elif category == 'Wisdom teeth detection':
                    is_impacted = item.get('is_impacted')
                    tooth_data[tooth_id].append({'is_wisdom_tooth': 'True'}) 
                    tooth_data[tooth_id].append({'is_impacted': is_impacted})
                else:
                    # for Dental caries / Periapical lesions / Historical treatments
                    box = item.get('box_2d')
                    label = item.get('label')
                    tooth_data[tooth_id].append({label: {'box_2d': box}})

    tooth_data = dict(tooth_data)

    for k, v in tooth_data.items():
        if len(v) == 1: # only have center points; invalid 
            continue
        question = choice(one_tooth_all_properties_Template).format(k.lower())
        formated_answer = "[\n" + ",\n".join( 
                                [str(json.dumps(item, ensure_ascii=False)) for item in v]
                            ) + "\n]"
        return_q_a_list.append(
            {
                "Question": question,
                "Answer": formated_answer
            }
        )

    return return_q_a_list


def extract_one_disease_all_teeth(historical_treatments_list):
    # 初始化结果字典
    label_to_entries = {}
    return_q_a_list = []
    # 遍历数据
    for entry in historical_treatments_list:
        label = entry['label']  # 获取 label
        # 如果 label 不在字典中，初始化为一个空列表
        if label not in label_to_entries:
            label_to_entries[label] = []
        # 将整条数据条目加入到对应的 label 列表中
        entry.pop('label')
        entry = dict(reversed(entry.items()))
        label_to_entries[label].append(entry)

    for k, v in label_to_entries.items():
        question = choice(one_disease_all_teeth_properties_Template).format(k.lower())
        formated_answer = "[\n" + ",\n".join(
                                [str(json.dumps(item, ensure_ascii=False)) for item in v]
                            ) + "\n]"
        return_q_a_list.append(
            {
                "Question": question,
                "Answer": formated_answer
            }
        )
    return return_q_a_list


def extract_all_Pathological_Findings(dict_data):
    diseases = ['caries', 'periapical lesions', 'impacted teeth', 'bone loss']

    # 初始化结果字典
    disease_to_elements = {disease: [] for disease in diseases}
    return_q_a_list = []
    # 遍历数据，根据需要提取相关的元素
    for category, elements in dict_data.items():
        for element in elements:
            # 检测疾病相关的特定关键词
            if 'label' in element:
                if element['label'] == 'Caries' or element['label'] == 'Deep caries':
                    disease_to_elements['caries'].append(element)
                elif element['label'] == 'Periapical lesions':
                    disease_to_elements['periapical lesions'].append(element)
                elif element['label'] == 'Bone loss':
                    disease_to_elements['bone loss'].append(element)
                elif element['label'] == 'Impacted tooth':
                    disease_to_elements['impacted teeth'].append(element)

            if 'is_impacted' in element and element['is_impacted'] == 'true':
                disease_to_elements['impacted teeth'].append(element)

    formated_answer = "[\n"
    flag_saved = False
    for k, v in disease_to_elements.items():
        if len(v) > 0:
            flag_saved = True
            formated_answer += ",\n".join(
                                    [str(json.dumps(item, ensure_ascii=False)) for item in v])
    if flag_saved:
        formated_answer += "\n]"
        question = choice(all_Pathological_Findings_Template)
        return_q_a_list.append(
            {
                "Question": question,
                "Answer": formated_answer
            }
        )
    return disease_to_elements, return_q_a_list


def extract_all_impacted_teeth(dict_data):
    all_impacted_teeth_data = dict_data['impacted teeth']
    return_q_a_list = []
    
    if len(all_impacted_teeth_data) > 0:
        question = choice(all_impacted_teeth_Template)
        formated_answer = "[\n" + ",\n".join(
                                [str(json.dumps(item, ensure_ascii=False)) for item in all_impacted_teeth_data]
                            ) + "\n]"
        return_q_a_list.append(
            {
                "Question": question,
                "Answer": formated_answer
            }
        )
    return return_q_a_list


def extract_all_jawbone_info(dict_data):
    # 初始化结果字典
    result = {'jawbone': []}
    return_q_a_list = []
    # 遍历输入数据
    for category, elements in dict_data.items():
        for element in elements:
            # 检查类别，并将相关元素添加到 'jawbone' 中
            if category.startswith('Bone loss detection'):
                result['jawbone'].append(element)
            elif category == 'Maxillary sinuses visibility':
                result['jawbone'].append(element)
            elif category == 'Mandibular canal visibility':  # 假如未来有这类数据
                result['jawbone'].append(element)

    if len(result['jawbone']) > 0:
        question = choice(all_jawbone_Template)
        formated_answer = "[\n" + ",\n".join(
                                [str(json.dumps(item, ensure_ascii=False)) for item in result['jawbone']]
                            ) + "\n]"
        return_q_a_list.append(
            {
                "Question": question,
                "Answer": formated_answer
            }
        )

    return return_q_a_list

def extract_field_from_jsons(input_folder, output_folder):
    """
    从输入文件夹中的所有JSON文件中提取指定字段，并保存到输出文件夹中的一个汇总JSON文件中
    
    :param input_folder: 包含输入JSON文件的文件夹路径
    :param output_folder: 输出文件夹路径
    :param field_name: 要提取的字段名，默认为'xxx'
    """
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(input_folder, filename)
            
            try:
                # 读取JSON文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                loc_caption = data['loc_caption'].split('including:\n')[1].strip()

                # for image aquisition_time 
                time_str = None
                if 'Panoramic Dental X-ray Imaging Time:' in loc_caption:
                    time_str = loc_caption.split('Panoramic Dental X-ray Imaging Time:')[1].split('\n')[0].strip()
                result = parse_medical_string(loc_caption)
                # print(result)
                q_a_pairs = []

                if time_str:
                    question = choice(Questions_Time_Acquisition_Template)
                    q_a_pairs.append(
                        {
                            "Question": question,
                            "Answer": time_str
                        }
                    )
                else:
                    if random.random() < 0.1:
                        q_a_pairs.append(
                            {
                                "Question": choice(Questions_Time_Acquisition_Template),
                                "Answer": choice(Questions_Time_Acquisition_Reject_Template),
                            }
                        )

                for category, value in result.items():
                    question = choice(Questions_Template).format(key_map[category])
                    formated_answer = "[\n" + ",\n".join(
                                [str(json.dumps(item, ensure_ascii=False)) for item in value]
                            ) + "\n]"
                    q_a_pairs.append(
                        {
                            "Question": question,
                            "Answer": formated_answer
                        }
                    )

                q_a_pairs += extract_one_tooth_all_properties(result)

                if 'Historical treatments' in result.keys():
                    q_a_pairs += extract_one_disease_all_teeth(result['Historical treatments'])

                disease_to_elements, return_q_a_list = extract_all_Pathological_Findings(result)
                q_a_pairs += return_q_a_list
                
                q_a_pairs += extract_all_impacted_teeth(disease_to_elements)

                q_a_pairs += extract_all_jawbone_info(result)

                # print(q_a_pairs)

                data.pop('properties')

                if 'sft_data' in data.keys():
                    data['sft_data']['loc_open_ended'] = q_a_pairs
                else:
                    data['sft_data'] = {}
                    data['sft_data']['loc_open_ended'] = q_a_pairs

                # 保存提取的数据到新JSON文件
                output_path = os.path.join(output_folder, filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    input_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report'
    output_folder = '/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/test_sft_open_json'
    
    extract_field_from_jsons(input_folder, output_folder)