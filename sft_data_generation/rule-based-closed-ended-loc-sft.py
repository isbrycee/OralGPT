import json
import os
import re
import random
from ast import literal_eval
import copy
from typing import List, Dict, Any, Tuple

# Question templates for different categories
COUNTING_TEMPLATES = [
    'How many teeth are visible in this panoramic dental X-ray image?',
    'What is the total number of visible teeth in this panoramic image?',
    'Count the number of teeth that can be seen in this dental X-ray.',
    'How many teeth with caries are detected in this panoramic image?',
    'What is the count of wisdom teeth visible in this X-ray?',
    'How many teeth show evidence of root canal treatment in this panoramic image?',
    'Count the number of missing teeth in this dental X-ray.',
    'How many areas of bone loss can be identified in this panoramic image?',
    'What is the total number of impacted teeth shown in this X-ray?',
    'How many dental fillings can be detected in this panoramic image?'
]

TOOTH_LOCALIZATION_TEMPLATES = [
    'Which tooth is located at coordinates [x, y] in the panoramic image?',
    'Which tooth appears within the bounding box [x1, y1, x2, y2] in this X-ray?',
    'Where is tooth #{} located in this panoramic dental image?',
    'Identify the position of tooth #{} in this X-ray image.',
    'Output the coordinates of tooth #{} in the panoramic dental X-ray.',
    'Detect the location of the {} in this panoramic image.',
    'Which tooth is positioned at the center point [{}, {}] in this dental X-ray?',
    'Locate tooth #{} in this panoramic radiograph.',
    'Where are the wisdom teeth positioned in this dental panoramic image?',
    'Identify the location of tooth #{} in this X-ray image.'
]

PATHOLOGY_TEMPLATES = [
    'Which teeth are affected by {} in this panoramic dental image?',
    'Where are the {} detected in this X-ray image?',
    'Output the positions of {} in the panoramic X-ray image.',
    'Which teeth show evidence of {} in this panoramic image?',
    'Identify the areas showing {} in the panoramic radiograph.',
    'Detect the locations of {} in this X-ray image.',
    'Which teeth have {} in this panoramic dental image?',
    'Where are the {} located in this X-ray?',
    'Identify which teeth exhibit {} in this panoramic image.',
    'Output the locations of all pathological findings in this dental X-ray.'
]

TREATMENT_TEMPLATES = [
    'Which teeth have {} in this panoramic dental image?',
    'Identify the teeth with {} in this X-ray.',
    'Where are {} detected in this panoramic image?',
    'Which teeth show evidence of {} in this X-ray?',
    'Output the positions of all dental restorations in the panoramic image.',
    'Detect the teeth with historical treatments in this dental X-ray.',
    'Which dental treatments are visible in the area [x1, y1, x2, y2]?',
    'Identify all teeth with {} in this panoramic radiograph.',
    'What type of dental treatment is shown in the bounding box [x1, y1, x2, y2]?',
    'Which teeth have undergone {} in this panoramic image?'
]

ANATOMY_TEMPLATES = [
    'Which anatomical structures are visible in the panoramic dental image?',
    'Where is the {} located in this X-ray image?',
    'Identify the {} in this panoramic radiograph.',
    'Detect the locations of bone structures in this dental X-ray.',
    'Output the positions of the {} in the panoramic image.',
    'Which anatomical landmarks have been accurately detected in the panoramic image?',
    'Please detect the {} in this X-ray image.',
    'Which areas show the {} in the panoramic image?',
    'Identify areas showing the {} in the panoramic radiograph.',
    'Output the boundaries of the {} in this dental X-ray.'
]

MISSING_TEETH_TEMPLATES = [
    'Which teeth are missing in this panoramic dental image?',
    'Identify the missing teeth in this X-ray.',
    'Where are gaps from missing teeth located in this panoramic image?',
    'Which areas show evidence of missing teeth in this dental X-ray?',
    'Output the positions of dental gaps in the panoramic image.',
    'Detect the missing teeth in the upper arch of this X-ray.',
    'Which teeth are absent in the lower jaw of this panoramic image?',
    'Identify all missing teeth positions in this dental radiograph.',
    'Which tooth numbers are missing in this panoramic X-ray?',
    'In which regions of the dental arches are teeth missing in this image?'
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
                
                entry_dict[key] = parsed_value
            
            if entry_dict:
                entries.append(entry_dict)
        
        result_dict[clean_category] = entries
    
    return result_dict

def generate_wrong_bbox(correct_bbox, image_width=1800, image_height=1000, other_bboxes=None):
    """Generate incorrect bounding boxes for multiple choice options"""
    if not correct_bbox or len(correct_bbox) != 4:
        return [0, 0, 100, 100]  # Default fallback
    
    x1, y1, x2, y2 = correct_bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    wrong_bbox_strategies = [
        # Translation errors
        lambda: [max(0, int(x1 + width * 0.3)), max(0, int(y1 + height * 0.3)), 
                 min(image_width, int(x2 + width * 0.3)), min(image_height, int(y2 + height * 0.3))],
        lambda: [max(0, int(x1 - width * 0.3)), max(0, int(y1 - height * 0.3)), 
                 min(image_width, int(x2 - width * 0.3)), min(image_height, int(y2 - height * 0.3))],
        lambda: [max(0, int(x1 + width * 0.4)), max(0, int(y1 - height * 0.2)), 
                 min(image_width, int(x2 + width * 0.4)), min(image_height, int(y2 - height * 0.2))],
        
        # Scaling errors
        lambda: [max(0, int(center_x - width * 0.7)), max(0, int(center_y - height * 0.7)), 
                 min(image_width, int(center_x + width * 0.7)), min(image_height, int(center_y + height * 0.7))],
        lambda: [max(0, int(center_x - width * 1.3)), max(0, int(center_y - height * 1.3)), 
                 min(image_width, int(center_x + width * 1.3)), min(image_height, int(center_y + height * 1.3))],
        
        # Contralateral location (horizontal mirror)
        lambda: [max(0, int(image_width - x2)), y1, min(image_width, int(image_width - x1)), y2],
        
        # Upper/lower correspondence (vertical shift)
        lambda: [x1, max(0, int(image_height - y2)), x2, min(image_height, int(image_height - y1))],
    ]
    
    # If other bboxes are provided, add them as potential wrong answers
    if other_bboxes:
        for bbox in other_bboxes[:3]:  # Limit to first 3 to avoid too many similar options
            if bbox != correct_bbox:
                wrong_bbox_strategies.append(lambda b=bbox: b)
    
    # Randomly select a strategy
    return random.choice(wrong_bbox_strategies)()

def generate_wrong_count(correct_count, max_possible=32):
    """Generate wrong count numbers for multiple choice options"""
    wrong_counts = set()
    
    # Strategy 1: correct count ±1
    candidates = [max(0, correct_count - 1), correct_count + 1]
    
    # Strategy 2: correct count ±2 or ±3
    candidates.extend([max(0, correct_count - 2), max(0, correct_count - 3), 
                      correct_count + 2, correct_count + 3])
    
    # Strategy 3: 0 and max possible value
    candidates.extend([0, max_possible])
    
    # Filter out the correct answer and select 3 wrong answers
    candidates = [c for c in candidates if c != correct_count]
    wrong_counts = random.sample(candidates, min(3, len(candidates)))
    
    # If we couldn't get 3 wrong answers, pad with more random numbers
    while len(wrong_counts) < 3:
        new_wrong = random.randint(0, max_possible)
        if new_wrong != correct_count and new_wrong not in wrong_counts:
            wrong_counts.add(new_wrong)
    
    return list(wrong_counts)

def generate_multiple_choice_options(correct_answer, wrong_options, shuffle=True):
    """Generate ABCD options with correct answer and wrong options"""
    all_options = [correct_answer] + wrong_options[:3]  # Limit to 3 wrong options
    
    if shuffle:
        random.shuffle(all_options)
    
    correct_index = all_options.index(correct_answer)
    correct_letter = chr(65 + correct_index)  # A, B, C, or D
    
    options = {
        "A": all_options[0],
        "B": all_options[1],
        "C": all_options[2],
        "D": all_options[3] if len(all_options) > 3 else "None of the above"
    }
    
    return options, correct_letter

def generate_counting_qa(parsed_data):
    """Generate counting questions and answers based on dental data"""
    qa_pairs = []
    
    # Teeth visibility counting
    if "Teeth visibility with center points" in parsed_data:
        teeth_count = len(parsed_data["Teeth visibility with center points"])
        if teeth_count > 0:
            question = random.choice(COUNTING_TEMPLATES[:3])
            wrong_counts = generate_wrong_count(teeth_count)
            options, correct_letter = generate_multiple_choice_options(teeth_count, wrong_counts)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {teeth_count} visible teeth in the panoramic dental X-ray image."
            })
    
    # Wisdom teeth counting
    if "Wisdom teeth detection" in parsed_data:
        wisdom_count = len(parsed_data["Wisdom teeth detection"])
        if wisdom_count > 0:
            question = COUNTING_TEMPLATES[4]  # "What is the count of wisdom teeth visible in this X-ray?"
            wrong_counts = generate_wrong_count(wisdom_count, max_possible=4)
            options, correct_letter = generate_multiple_choice_options(wisdom_count, wrong_counts)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {wisdom_count} wisdom teeth visible in the X-ray image."
            })
    
    # Caries counting
    if "Dental caries detection" in parsed_data:
        caries_count = len(parsed_data["Dental caries detection"])
        if caries_count > 0:
            question = COUNTING_TEMPLATES[3]  # "How many teeth with caries are detected in this panoramic image?"
            wrong_counts = generate_wrong_count(caries_count)
            options, correct_letter = generate_multiple_choice_options(caries_count, wrong_counts)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {caries_count} teeth with caries detected in the panoramic image."
            })
    
    # Missing teeth counting
    if "Missing teeth detection" in parsed_data:
        missing_count = len(parsed_data["Missing teeth detection"])
        if missing_count > 0:
            question = COUNTING_TEMPLATES[6]  # "Count the number of missing teeth in this dental X-ray."
            wrong_counts = generate_wrong_count(missing_count)
            options, correct_letter = generate_multiple_choice_options(missing_count, wrong_counts)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {missing_count} missing teeth in the dental X-ray."
            })
    
    # Bone loss counting
    if "Bone loss detection" in parsed_data:
        bone_loss_count = len(parsed_data["Bone loss detection"])
        if bone_loss_count > 0:
            question = COUNTING_TEMPLATES[7]  # "How many areas of bone loss can be identified in this panoramic image?"
            wrong_counts = generate_wrong_count(bone_loss_count, max_possible=4)
            options, correct_letter = generate_multiple_choice_options(bone_loss_count, wrong_counts)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {bone_loss_count} areas of bone loss identified in the panoramic image."
            })
    
    # Treatments counting
    if "Historical treatments" in parsed_data:
        treatment_count = len(parsed_data["Historical treatments"])
        if treatment_count > 0:
            question = COUNTING_TEMPLATES[9]  # "How many dental fillings can be detected in this panoramic image?"
            wrong_counts = generate_wrong_count(treatment_count)
            options, correct_letter = generate_multiple_choice_options(treatment_count, wrong_counts)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {treatment_count} dental treatments detected in the panoramic image."
            })
    
    return qa_pairs

def generate_tooth_localization_qa(parsed_data):
    """Generate tooth localization questions and answers"""
    qa_pairs = []
    
    # Only proceed if we have tooth data
    if "Teeth visibility with center points" not in parsed_data or not parsed_data["Teeth visibility with center points"]:
        return qa_pairs
    
    teeth_data = parsed_data["Teeth visibility with center points"]
    
    # Sample a few teeth to ask about
    sampled_teeth = random.sample(teeth_data, min(3, len(teeth_data)))
    
    for tooth in sampled_teeth:
        if "tooth_id" not in tooth or "point_2d" not in tooth:
            continue
            
        tooth_id = tooth["tooth_id"]
        x, y = tooth["point_2d"]
        
        # Generate a question about tooth location
        template = random.choice([t for t in TOOTH_LOCALIZATION_TEMPLATES if '{}' in t])
        
        if "center point" in template:
            question = template.format(x, y)
            correct_answer = tooth_id
            
            # Generate wrong options (other tooth IDs)
            wrong_options = [t["tooth_id"] for t in teeth_data if t["tooth_id"] != tooth_id]
            if len(wrong_options) < 3:
                # Add some random tooth numbers if we don't have enough
                potential_ids = [str(i) for i in range(11, 49) if str(i) not in wrong_options and str(i) != tooth_id]
                wrong_options.extend(random.sample(potential_ids, min(3 - len(wrong_options), len(potential_ids))))
            
            options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"The tooth at center point [{x}, {y}] is tooth #{tooth_id}."
            })
        else:
            # Question asking about the location of a specific tooth
            question = template.format(tooth_id)
            
            # For these questions, the answer would be the bounding box
            # Since we only have center points, create a small bounding box around it
            correct_bbox = [x - 20, y - 20, x + 20, y + 20]
            
            # Generate wrong bounding boxes
            other_center_points = [t["point_2d"] for t in teeth_data if t["tooth_id"] != tooth_id]
            other_bboxes = [[p[0] - 20, p[1] - 20, p[0] + 20, p[1] + 20] for p in other_center_points]
            
            wrong_bboxes = []
            for _ in range(3):
                wrong_bboxes.append(generate_wrong_bbox(correct_bbox, other_bboxes=other_bboxes))
            
            # Format bounding boxes as strings for display
            correct_bbox_str = str(correct_bbox)
            wrong_bbox_strs = [str(bbox) for bbox in wrong_bboxes]
            
            options, correct_letter = generate_multiple_choice_options(correct_bbox_str, wrong_bbox_strs)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"Tooth #{tooth_id} is located at the bounding box {correct_bbox_str} in the panoramic dental image."
            })
    
    return qa_pairs

def generate_pathology_qa(parsed_data):
    """Generate questions about pathological conditions"""
    qa_pairs = []
    
    # Dental caries questions
    if "Dental caries detection" in parsed_data and parsed_data["Dental caries detection"]:
        caries_data = parsed_data["Dental caries detection"]
        
        # Get affected tooth IDs
        caries_teeth = [item.get("tooth_id") for item in caries_data if "tooth_id" in item]
        caries_teeth = [t for t in caries_teeth if t != "unknown"]
        
        if caries_teeth:
            # Question about which teeth have caries
            question = random.choice([t.format("caries") for t in PATHOLOGY_TEMPLATES[:4]])
            correct_answer = ", ".join([f"#{t}" for t in caries_teeth])
            
            # Generate wrong options
            if "Teeth visibility with center points" in parsed_data:
                all_teeth = [t.get("tooth_id") for t in parsed_data["Teeth visibility with center points"]]
                non_caries_teeth = [t for t in all_teeth if t not in caries_teeth]
                
                wrong_options = []
                for _ in range(3):
                    if non_caries_teeth:
                        sample_size = min(len(caries_teeth), len(non_caries_teeth))
                        wrong_sample = random.sample(non_caries_teeth, sample_size)
                        wrong_options.append(", ".join([f"#{t}" for t in wrong_sample]))
                    else:
                        wrong_options.append("No teeth affected")
            else:
                wrong_options = ["No teeth affected", "All teeth affected", "Cannot be determined"]
            
            options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"The teeth affected by caries are: {correct_answer}."
            })
    
    # Periapical lesions questions
    if "Periapical lesions detection" in parsed_data and parsed_data["Periapical lesions detection"]:
        lesion_data = parsed_data["Periapical lesions detection"]
        
        question = random.choice([t.format("periapical lesions") for t in PATHOLOGY_TEMPLATES[4:8]])
        
        if lesion_data:
            # For bounding box questions
            sample_lesion = random.choice(lesion_data)
            if "box_2d" in sample_lesion:
                correct_bbox = sample_lesion["box_2d"]
                correct_bbox_str = str(correct_bbox)
                
                # Generate wrong bounding boxes
                wrong_bboxes = []
                for _ in range(3):
                    wrong_bboxes.append(generate_wrong_bbox(correct_bbox))
                
                wrong_bbox_strs = [str(bbox) for bbox in wrong_bboxes]
                
                options, correct_letter = generate_multiple_choice_options(correct_bbox_str, wrong_bbox_strs)
                
                qa_pairs.append({
                    "question": question,
                    "options": options,
                    "answer": correct_letter,
                    "explanation": f"Periapical lesions are detected at the bounding box {correct_bbox_str}."
                })
    
    return qa_pairs

def generate_treatment_qa(parsed_data):
    """Generate questions about dental treatments"""
    qa_pairs = []
    
    if "Historical treatments" in parsed_data and parsed_data["Historical treatments"]:
        treatments_data = parsed_data["Historical treatments"]
        
        # Group treatments by type
        treatment_types = {}
        for treatment in treatments_data:
            if "label" in treatment:
                label = treatment["label"]
                if label not in treatment_types:
                    treatment_types[label] = []
                treatment_types[label].append(treatment)
        
        # Generate questions for each treatment type
        for treatment_type, items in treatment_types.items():
            if items:
                question = random.choice([t.format(treatment_type.lower()) for t in TREATMENT_TEMPLATES[:4]])
                
                # Get affected tooth IDs
                treatment_teeth = [item.get("tooth_id") for item in items if "tooth_id" in item]
                treatment_teeth = [t for t in treatment_teeth if t != "unknown"]
                
                if treatment_teeth:
                    correct_answer = ", ".join([f"#{t}" for t in treatment_teeth])
                    
                    # Generate wrong options
                    if "Teeth visibility with center points" in parsed_data:
                        all_teeth = [t.get("tooth_id") for t in parsed_data["Teeth visibility with center points"]]
                        non_treated_teeth = [t for t in all_teeth if t not in treatment_teeth]
                        
                        wrong_options = []
                        for _ in range(3):
                            if non_treated_teeth:
                                sample_size = min(len(treatment_teeth), len(non_treated_teeth))
                                wrong_sample = random.sample(non_treated_teeth, sample_size)
                                wrong_options.append(", ".join([f"#{t}" for t in wrong_sample]))
                            else:
                                wrong_options.append("No teeth affected")
                    else:
                        wrong_options = ["No teeth affected", "All teeth affected", "Cannot be determined"]
                    
                    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                    
                    qa_pairs.append({
                        "question": question,
                        "options": options,
                        "answer": correct_letter,
                        "explanation": f"The teeth with {treatment_type.lower()} are: {correct_answer}."
                    })
                
                # Also generate a specific bounding box question
                sample_treatment = random.choice(items)
                if "box_2d" in sample_treatment:
                    question = TREATMENT_TEMPLATES[8]  # "What type of dental treatment is shown in the bounding box...?"
                    question = question.replace("[x1, y1, x2, y2]", str(sample_treatment["box_2d"]))
                    
                    correct_answer = treatment_type
                    
                    # Generate wrong options (other treatment types or made-up ones)
                    standard_treatments = ["Filling", "Crown", "Root canal treatment", "Implant", "Extraction", "Bridge"]
                    wrong_options = [t for t in standard_treatments if t != treatment_type]
                    
                    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                    
                    qa_pairs.append({
                        "question": question,
                        "options": options,
                        "answer": correct_letter,
                        "explanation": f"The dental treatment shown in the specified bounding box is {treatment_type}."
                    })
    
    return qa_pairs

def generate_anatomy_qa(parsed_data):
    """Generate questions about anatomical structures"""
    qa_pairs = []
    
    # Mandibular canal questions
    if "Mandibular canal visibility" in parsed_data and parsed_data["Mandibular canal visibility"]:
        canal_data = parsed_data["Mandibular canal visibility"]
        
        if canal_data:
            question = random.choice([t.format("mandibular canal") for t in ANATOMY_TEMPLATES[1:5]])
            
            # For bounding box questions
            if len(canal_data) == 2:  # Both left and right canals visible
                correct_answer = "Both left and right sides"
                wrong_options = ["Left side only", "Right side only", "Not visible"]
            else:
                sample_canal = random.choice(canal_data)
                if "box_2d" in sample_canal:
                    correct_bbox = sample_canal["box_2d"]
                    correct_bbox_str = str(correct_bbox)
                    
                    # Generate wrong bounding boxes
                    wrong_bboxes = []
                    for _ in range(3):
                        wrong_bboxes.append(generate_wrong_bbox(correct_bbox))
                    
                    wrong_bbox_strs = [str(bbox) for bbox in wrong_bboxes]
                    
                    options, correct_letter = generate_multiple_choice_options(correct_bbox_str, wrong_bbox_strs)
                    
                    qa_pairs.append({
                        "question": question,
                        "options": options,
                        "answer": correct_letter,
                        "explanation": f"The mandibular canal is located at the bounding box {correct_bbox_str}."
                    })
    
    # Maxillary sinus questions
    if "Maxillary sinuses visibility" in parsed_data and parsed_data["Maxillary sinuses visibility"]:
        sinus_data = parsed_data["Maxillary sinuses visibility"]
        
        if sinus_data:
            question = random.choice([t.format("maxillary sinuses") for t in ANATOMY_TEMPLATES[5:]])
            
            # For yes/no questions
            if sinus_data:
                correct_answer = "Clearly visible"
                wrong_options = ["Not visible", "Partially visible", "Cannot be determined"]
                
                options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                
                qa_pairs.append({
                    "question": question,
                    "options": options,
                    "answer": correct_letter,
                    "explanation": f"The maxillary sinuses are clearly visible in the panoramic image."
                })
    
    return qa_pairs

def generate_missing_teeth_qa(parsed_data):
    """Generate questions about missing teeth"""
    qa_pairs = []
    
    if "Missing teeth detection" in parsed_data and parsed_data["Missing teeth detection"]:
        missing_data = parsed_data["Missing teeth detection"]
        
        if missing_data:
            question = random.choice(MISSING_TEETH_TEMPLATES[:4])
            
            # Group by side (upper/lower)
            sides = {}
            for item in missing_data:
                if "side" in item:
                    side = item["side"]
                    if side not in sides:
                        sides[side] = []
                    sides[side].append(item)
            
            if sides:
                if len(sides) > 1:  # Both upper and lower missing teeth
                    correct_answer = "Both upper and lower jaw"
                    wrong_options = ["Upper jaw only", "Lower jaw only", "No missing teeth"]
                else:
                    side = list(sides.keys())[0]
                    correct_answer = f"{side.capitalize()} jaw"
                    wrong_options = [
                        "Both upper and lower jaw",
                        "Upper jaw" if side.lower() == "lower" else "Lower jaw",
                        "No missing teeth"
                    ]
                
                options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                
                qa_pairs.append({
                    "question": question,
                    "options": options,
                    "answer": correct_letter,
                    "explanation": f"Missing teeth are detected in the {correct_answer.lower()}."
                })
    
    return qa_pairs

def generate_qa_for_json(data):
    """Generate all types of QA pairs for a JSON file"""
    all_qa_pairs = []
    
    # Parse the localization caption
    if "loc_caption" in data:
        try:
            loc_caption = data["loc_caption"].split('including:\n')[1].strip()
            parsed_data = parse_medical_string(loc_caption)
            
            # Generate different types of QA pairs
            all_qa_pairs.extend(generate_counting_qa(parsed_data))
            all_qa_pairs.extend(generate_tooth_localization_qa(parsed_data))
            all_qa_pairs.extend(generate_pathology_qa(parsed_data))
            all_qa_pairs.extend(generate_treatment_qa(parsed_data))
            all_qa_pairs.extend(generate_anatomy_qa(parsed_data))
            all_qa_pairs.extend(generate_missing_teeth_qa(parsed_data))
            
            # Shuffle and limit to a reasonable number (e.g., 10)
            random.shuffle(all_qa_pairs)
            all_qa_pairs = all_qa_pairs[:10]
        except Exception as e:
            print(f"Error generating QA pairs: {str(e)}")
    
    return all_qa_pairs

def process_json_files(input_folder, output_folder):
    """Process all JSON files in the input folder and add QA data"""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each JSON file
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Read input JSON
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Generate QA pairs
                qa_pairs = generate_qa_for_json(data)
                
                # Add QA pairs to the data
                if "sft_data" not in data:
                    data["sft_data"] = {}
                
                data["sft_data"]["loc_closed_ended"] = qa_pairs
                
                # Write updated JSON to output folder
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"Processed {filename}: Added {len(qa_pairs)} QA pairs")
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define input and output folders
    input_folder = "input_jsons/"
    output_folder = "output_jsons/"
    
    # Process all JSON files
    process_json_files(input_folder, output_folder)
