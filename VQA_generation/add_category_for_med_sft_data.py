import json
import os
import re
from typing import Dict, List


# 分类配置（保持优先级顺序）
CATEGORY_ORDER = ['teeth', 'patho', 'his', 'jaw', 'summ']
CATEGORY_PATTERNS = {
    'teeth': r'\b(teeth|tooth|wisdom|missing|impacted|erupted|status|dentition)\b',
    'patho': r'\b(carious|suspected|patho|caries|lesion|cyst|periapical|pathological|finding)\b',
    'his': r'\b(intervention|filling|crown|implant|restoration|root canal|historical|fillings|crowns|implants|interventions|visible|restorations)\b',
    'jaw': r'\b(jaws|bones|jaw|bone loss|mandibular|maxillary|sinus|bone|visible structures|visible bilaterally|structure|mandible|jawbone|sinuses|structures|jawbones|visible)\b',
    'summ': r'\b(surgical|immediate|needs|investigation|monitored|attention|advised|summary|recommendation|concern|measure|recommended|needed|clinical|priority|suggestion|preventive|evaluation|preventive|follow-up|monitoring)\b'
}

def validate_json_structure(data: Dict, filename: str) -> bool:
    """验证JSON文件结构"""
    required_keys = {
        'vqa_data': {
            'Closed-End Questions': list,
            'Open-End Questions': list
        }
    }
    
    try:
        # 检查一级结构
        if 'vqa_data' not in data:
            raise KeyError(f"缺少顶级字段 'vqa_data'")
        
        sft_data = data['vqa_data']
        
        # 检查二级结构
        for section in ['med_closed_ended', 'med_open_ended']:
            if section not in sft_data:
                raise KeyError(f"缺少问题区块 '{section}'")
            if not isinstance(sft_data[section], list):
                raise TypeError(f"'{section}' 应为列表类型")
            
        # 检查问题结构
        for section in ['med_closed_ended', 'med_open_ended']:
            for i, q in enumerate(sft_data[section]):
                if 'Question' not in q or 'Answer' not in q:
                    raise KeyError(f"{section} 第{i+1}个问题缺少必要字段")
                
        return True
    except (KeyError, TypeError) as e:
        print(f"[结构异常] {filename}: {str(e)}")
        return False


def classify_question(question: str) -> str:
    """多类别分类逻辑"""
    question = question.lower()
    matches = set()
    
    # 同时匹配所有类别
    for cat, pattern in CATEGORY_PATTERNS.items():
        # print(pattern)
        # print(question)
        if re.search(pattern, question):
            matches.add(cat)
    
    # 按优先级排序并去重
    ordered = [cat for cat in CATEGORY_ORDER if cat in matches]
    # print(ordered)
    # 默认逻辑：当完全无匹配时
    if not ordered:
        # 根据问题类型设置默认值
        # if any(w in question for w in ["list", "describe"]):
        #     return "summ"
        print(question)
        print("not matched!")
        return "teeth"  # 最通用的默认值
    
    return ",".join(ordered)

def process_json_data(data: Dict) -> Dict:
    """增强型数据处理"""
    def process_qa(qa_list: List[Dict]):
        for qa in qa_list:
            # 保留原始小写处理
            orig_question = qa["Question"].lower()
            qa["category"] = classify_question(orig_question)
            # 特殊处理：当问题包含多个实体时
            if " and " in orig_question:
                if 'jaw' in qa["category"] and not re.search(r'\bjaw\b', orig_question):
                    qa["category"] = qa["category"].replace('jaw,', '')
    
    process_qa(data['vqa_data']['med_closed_ended'])
    process_qa(data['vqa_data']['med_open_ended'])
    return data


def process_folder(input_dir: str, output_dir: str):
    """增强型文件夹处理"""
    os.makedirs(output_dir, exist_ok=True)
    error_log = []
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # 读取文件
            with open(input_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError("无效的JSON格式")
            
            # 结构验证
            if not validate_json_structure(data, filename):
                error_log.append(filename)
                continue
                
            # 数据处理
            processed_data = process_json_data(data)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            error_log.append(filename)
            print(f"[处理失败] {filename}: {str(e)}")
    
    # 输出错误报告
    if error_log:
        print("\n" + "="*40)
        print(f"共发现 {len(error_log)} 个异常文件：")
        for f in error_log:
            print(f"  - {f}")
        print("="*40)

# 使用示例
if __name__ == "__main__":
    input_directory = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/MM-Oral-OPG-vqa-loc-med"  # 替换为实际输入路径
    output_directory = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/MM-Oral-OPG-vqa-loc-med-output"  # 替换为实际输出路径
    
    process_folder(input_directory, output_directory)
    print(f"处理完成！文件已保存到 {output_directory}")