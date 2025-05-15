import json
import os
import re
import random
from ast import literal_eval
from typing import Dict, List, Optional, Union, Set, Tuple

# 明确定义治疗类型常量
TREATMENT_TYPES = ["Filling", "Crown", "Root canal treatment", "Implant"]

TEMPLATE_DICT = {
    "Teeth visibility with center points": {
        "count": [
            "How many teeth are visible in this panoramic dental X-ray image?",
            "What is the total number of visible teeth in this panoramic image?",
            "Count the number of teeth that can be seen in this dental X-ray."
        ],
        "spatial": [
            # 从坐标到牙齿ID的模板
            "Which tooth is located at coordinates [{point_2d[0]}, {point_2d[1]}] in the panoramic image?",
            "What tooth can be found at position [{point_2d[0]}, {point_2d[1]}] in this X-ray?",
            "Identify the tooth located at coordinates [{point_2d[0]}, {point_2d[1]}] in this dental X-ray.",
            
            # 从牙齿ID到坐标的模板
            "What are the coordinates of tooth #{tooth_id} in this panoramic dental X-ray?",
            "Where is tooth #{tooth_id} located in this panoramic X-ray? Give the exact coordinates.",
            "Please provide the center point coordinates of tooth #{tooth_id} in this dental panoramic X-ray.",
            "Can you locate tooth #{tooth_id} in this image? Please provide its coordinates.",
            "What are the center point coordinates of tooth #{tooth_id} in this X-ray image?",
            
            # 通用模板（可能需要在代码中特殊处理）
            "What object is present at coordinates [{point_2d[0]}, {point_2d[1]}]?",
            "What tooth number corresponds to the point at [{point_2d[0]}, {point_2d[1]}]?",
            "Give the ID number of the tooth at position [{point_2d[0]}, {point_2d[1]}]."
        ]
    },
    "Wisdom teeth detection": {
        "count": [
            "What is the count of wisdom teeth visible in this X-ray?",
            "How many wisdom teeth can be identified in this panoramic image?",
            "Are any impacted wisdom teeth present in this panoramic dental X-ray? If so, how many?"
        ],
        "spatial": [
            "Where is wisdom tooth #{tooth_id} located in this panoramic X-ray? Provide the bounding box coordinates.",
            "What is the location of the impacted wisdom tooth in this X-ray? Give the exact bbox coordinates.",
            "What are the box_2d coordinates of wisdom tooth #{tooth_id} in this panoramic dental X-ray?",
            "If impacted wisdom teeth are present, please provide their locations and box coordinates.", 
            "Draw a bounding box around wisdom tooth #{tooth_id}. What are the coordinates?",
            "Locate and provide the exact coordinates for wisdom tooth #{tooth_id} in this panoramic X-ray.",
            "What object is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what object is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What is it in this region {box_2d}?",
            "What object is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what object can be found?",
            "Can you identify the object within the bounding box {box_2d}?",
            "What object is present in this region {box_2d}?"
        ]
    },
    "Missing teeth detection": {
        "spatial": [
            "What are the locations of missing teeth in this panoramic X-ray? Specify the bbox coordinates for each.",
            "What are the box_2d coordinates of the missing tooth area in this panoramic dental X-ray?",
            "Please provide the locations of all missing teeth in this panoramic X-ray as bounding box coordinates.",
            "Mark the missing tooth areas with bounding boxes. What are their coordinates?",
            "Where exactly are the missing teeth located? Provide precise bbox coordinates.",
            "What condition is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what condition is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What condition is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what condition can be found?",
            "Can you identify the condition within the bounding box {box_2d}?",
            "What condition is present in this region {box_2d}?"
        ]
    },
    "Non-wisdom impacted teeth detection": {
        "count": [
            "How many non-wisdom impacted teeth are visible in this panoramic X-ray?",
        ],
        "spatial": [
            "Where is tooth #{tooth_id} impacted in this panoramic X-ray? Give the exact bbox coordinates.",
            "Which teeth are impacted (excluding wisdom teeth) in this dental image? Provide bbox coordinates for each.",
            "What are the box_2d coordinates of the impacted non-wisdom tooth #{tooth_id} in this panoramic dental X-ray?",
            "Please specify the locations of all impacted non-wisdom teeth and provide their box coordinates.",
            "Draw a bounding box around the impacted tooth #{tooth_id}. What are its coordinates?",
            "Mark each impacted non-wisdom tooth with a bounding box. Provide the coordinates.",
            "What object is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what object is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What is it in this region {box_2d}?",
            "What object is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what object can be found?",
            "Can you identify the object within the bounding box {box_2d}?",
            "What object is present in this region {box_2d}?"
        ]
    },
    "Dental caries detection": {
        "count": [
            "How many teeth with caries are detected in this panoramic image?",
        ],
        "spatial": [
            "Which teeth are affected by caries in this panoramic dental image? Give bbox coordinates for each caries.",
            "Where is the caries located on tooth #{tooth_id} in this X-ray? Provide exact bbox coordinates.",
            "What are the box_2d coordinates of the caries on tooth #{tooth_id} in this panoramic dental X-ray?",
            "Please identify which teeth have caries and provide their box coordinates.",
            "Mark each caries with a bounding box. What are the coordinates?",
            "Draw a bounding box around the caries on tooth #{tooth_id}. What are its coordinates?",
            "What disease is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what disease is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What disease is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what disease can be found?",
            "Can you identify the disease within the bounding box {box_2d}?",
            "What disease is present in this region {box_2d}?"
        ]
    },
    "Periapical lesions detection": {
        "count": [
            "How many teeth show periapical lesions in this panoramic X-ray?",
        ],
        "type": [
            "What type of periapical lesion is detected on tooth #{tooth_id} in this X-ray?",
            "What is the specific type of periapical lesion associated with tooth #{tooth_id}?",
            "Can you identify the type of periapical lesion present on tooth #{tooth_id}?",
            "What kind of periapical lesion can be observed on tooth #{tooth_id} in this X-ray?"
        ],
        "spatial": [
            "Where is the periapical lesion located in relation to tooth #{tooth_id}? Give precise bbox coordinates.",
            "Which teeth are affected by periapical lesions in this panoramic X-ray? Mark each with bbox coordinates.",
            "What are the box_2d coordinates of the periapical lesion associated with tooth #{tooth_id} in this panoramic dental X-ray?",
            "Please identify all periapical lesions and provide their box coordinates.",
            "Draw a bounding box around each periapical lesion. What are their coordinates?",
            "Mark the periapical lesion associated with tooth #{tooth_id} with a bounding box. What are its coordinates?",
            "What disease is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what disease is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What disease is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what disease can be found?",
            "Can you identify the disease within the bounding box {box_2d}?",
            "What disease is present in this region {box_2d}?"
        ]
    },
    # {treatment_type} includes：['filling', 'crown', 'implant', 'root canal treatments']
    "Historical treatments": {
        "types": [
            "What types of dental treatments are visible in this panoramic X-ray?"
        ],
        "count": [
            "How many dental fillings can be detected in this panoramic image?",
            "How many root canal treatments can be detected in this panoramic image?",
            "How many dental crowns can be detected in this panoramic image?",
            "How many dental implants can be detected in this panoramic image?",
        ],
        "spatial": [
            "Which teeth have dental crowns in this panoramic dental image? Provide bbox coordinates for each crown.",
            "Which teeth show evidence of dental implants in this panoramic image? Mark each with a bounding box.",
            "Where are dental restorations detected in this panoramic image? Give exact bbox coordinates.",
            "What are the box_2d coordinates of the {treatment_type} on tooth #{tooth_id} in this panoramic dental X-ray?",
            "Please indicate which teeth have root canal treatments and provide their box coordinates.",
            "Draw a bounding box around the {treatment_type} on tooth #{tooth_id}. What are its coordinates?",
            "Mark each {treatment_type} with a bounding box. What are their coordinates?",
            "What historical treatment is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what historical dental treatment is present?",
            "Do you know what historical treatment is in the bounding box {box_2d}?",
            "What historical dental treatment is present in this region {box_2d}?",
            "What historical treatment is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what historical dental treatment can be found?",
            "Can you identify the historical dental treatment within the bounding box {box_2d}?",
            "What historical dental treatment is present in this region {box_2d}?"
        ],
        "tooth_based": [
            "What historical treatment is present on tooth #{tooth_id}?",
            "What type of dental restoration is found on tooth #{tooth_id}?",
            "What previous dental treatment has been done on tooth #{tooth_id}?",
            "Can you identify the dental treatment present on tooth #{tooth_id}?",
            "What dental intervention has been performed on tooth #{tooth_id} in the past?",
            "What historical dental intervention is visible on tooth #{tooth_id}?"
        ]
    },
    "Bone loss detection": {
        "spatial": [
            "Where is bone loss detected in this panoramic X-ray? Provide bbox coordinates for each area.",
            "What are the box_2d coordinates of the bone loss area in this panoramic dental X-ray?",
            "Please specify the location of all bone loss areas and provide their box coordinates.",
            "Draw a bounding box around each area of bone loss. What are their coordinates?",
            "Mark the regions of bone loss with bounding boxes. What are the exact coordinates?",
            "What phenomenon is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what phenomenon is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What is it in this region {box_2d}?",
            "What phenomenon is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what phenomenon can be found?",
            "Can you identify the phenomenon within the bounding box {box_2d}?",
            "What phenomenon is present in this region {box_2d}?"
        ]
    },
    "Mandibular canal visibility": {
        "spatial": [
            "Where is the mandibular canal located in this panoramic X-ray? Provide precise bbox coordinates.",
            "How clear is the visualization of the mandibular canal in this image? Mark it with a bounding box.",
            "What are the box_2d coordinates of the mandibular canal in this panoramic dental X-ray?",
            "If the mandibular canal is visible, please describe its appearance and provide its box coordinates.",
            "Draw a bounding box around the mandibular canal. What are its coordinates?",
            "Mark the mandibular canal with a bounding box. What are the exact coordinates?",
            "What structure is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what structure is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What structure is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what structure can be found?",
            "Can you identify the structure within the bounding box {box_2d}?",
            "What structure is present in this region {box_2d}?"
        ]
    },
    "Maxillary sinuses visibility": {
        "spatial": [
            "Where are the maxillary sinuses located in this panoramic X-ray? Provide bbox coordinates for each.",
            "How well can you visualize the maxillary sinuses in this panoramic image? Mark them with bounding boxes.",
            "What are the box_2d coordinates of the maxillary sinus on the {side} side in this panoramic dental X-ray?",
            "If the maxillary sinuses are visible, please describe their appearance and provide their box coordinates.",
            "Draw a bounding box around each maxillary sinus. What are their coordinates?",
            "Mark the maxillary sinuses with bounding boxes. What are the exact coordinates?",
            "What structure is present within the coordinates {box_2d}?",
            "Within the specified region {box_2d}, what structure is present?",
            "Do you know what it is in the bounding box {box_2d}?",
            "What structure is located within the coordinates {box_2d}?",
            "Within the specified area {box_2d}, what structure can be found?",
            "Can you identify the structure within the bounding box {box_2d}?",
            "What structure is present in this region {box_2d}?"
        ]
    }
}

# 更新关键词字典，添加具体治疗类型的关键词
CATEGORY_KEYWORDS = {
    "Teeth visibility with center points": ["teeth", "tooth", "visible"],
    "Wisdom teeth detection": ["wisdom"],
    "Missing teeth detection": ["missing"],
    "Non-wisdom impacted teeth detection": ["impacted", "non-wisdom"],
    "Dental caries detection": ["caries", "decay", "cavities"],
    "Periapical lesions detection": ["periapical", "lesion"],
    "Historical treatments": ["treatment", "historical treatment", "dental treatment"],
    "Dental fillings": ["filling", "fillings", "dental filling"],
    "Dental crowns": ["crown", "crowns", "dental crown"],
    "Root canal treatments": ["root canal"],
    "Dental implants": ["implant", "implants", "dental implant"],
    "Bone loss detection": ["bone loss"],
    "Mandibular canal visibility": ["mandibular", "canal"],
    "Maxillary sinuses visibility": ["maxillary", "sinus", "sinuses"]
}

# 特殊条件列表，更新为包含具体治疗类型
SPECIAL_CONDITIONS = [
    "Dental caries detection", 
    "Periapical lesions detection", 
    "Historical treatments",
    "Bone loss detection",
    "Mandibular canal visibility",
    "Maxillary sinuses visibility",
    "Wisdom teeth detection",
    "Missing teeth detection",
    "Non-wisdom impacted teeth detection"
]

def parse_medical_string(input_str: str) -> Dict:
    """Parse medical detection string to structured dictionary"""
    pattern = r"""
        ([A-Za-z\s]+)\s+    # Match category name
        (?:\(.*?\):\s*)     # Skip parenthesized content
        (\[.*?\])\s*        # Capture detection items
        (?=\n\S|\Z)         # Lookahead for end position
    """
    
    result_dict = {}
    
    # Split main categories
    for match in re.finditer(pattern, input_str, re.VERBOSE | re.DOTALL):
        category, entries_str = match.groups()
        
        # Clean category name
        clean_category = re.sub(r'\s+', ' ', category).strip()
        
        # Parse entries
        entries = []
        entry_pattern = r"\{([^}]+)\}"
        for entry_match in re.finditer(entry_pattern, entries_str):
            entry_str = entry_match.group(1)
            
            # Convert key-value pairs
            entry_dict = {}
            kv_pattern = r"'(\w+)':\s*(.+?)(?=,\s*'|$)"
            for kv in re.finditer(kv_pattern, entry_str):
                key, value = kv.groups()
                
                # Value type conversion
                try:
                    parsed_value = literal_eval(value.strip())
                except:
                    parsed_value = value.strip()
                
                entry_dict[key] = parsed_value
            
            if entry_dict:
                entries.append(entry_dict)
        
        result_dict[clean_category] = entries
    
    return result_dict

def generate_wrong_bbox(correct_bbox: List[int], image_width: int = 1800, 
                       image_height: int = 1000, other_bboxes: Optional[List[List[int]]] = None,
                       existing_wrong_bboxes: Optional[List[List[int]]] = None) -> List[int]:
    """Generate incorrect bounding boxes with much larger differences for multiple choice options"""
    if not correct_bbox or len(correct_bbox) != 4:
        return [0, 0, 100, 100]  # Default fallback
    
    if existing_wrong_bboxes is None:
        existing_wrong_bboxes = []
    
    x1, y1, x2, y2 = correct_bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Greatly enhanced strategies with much larger displacements
    wrong_bbox_strategies = [
        # Major translation errors (100-300% displacement)
        lambda: [max(0, int(x1 + width * 2.0)), max(0, int(y1 + height * 2.0)), 
                 min(image_width, int(x2 + width * 2.0)), min(image_height, int(y2 + height * 2.0))],
        lambda: [max(0, int(x1 - width * 2.5)), max(0, int(y1 - height * 2.5)), 
                 min(image_width, int(x2 - width * 2.5)), min(image_height, int(y2 - height * 2.5))],
        lambda: [max(0, int(x1 + width * 1.8)), max(0, int(y1 - height * 1.8)), 
                 min(image_width, int(x2 + width * 1.8)), min(image_height, int(y2 - height * 1.8))],
        
        # Extreme scaling errors (25-400% size change)
        lambda: [max(0, int(center_x - width * 0.25)), max(0, int(center_y - height * 0.25)), 
                 min(image_width, int(center_x + width * 0.25)), min(image_height, int(center_y + height * 0.25))],
        lambda: [max(0, int(center_x - width * 4.0)), max(0, int(center_y - height * 4.0)), 
                 min(image_width, int(center_x + width * 4.0)), min(image_height, int(center_y + height * 4.0))],
        
        # Contralateral location (horizontal mirror with larger offset)
        lambda: [max(0, int(image_width - x2 - width * 0.5)), max(0, int(y1 - height * 0.5)), 
                 min(image_width, int(image_width - x1 + width * 0.5)), min(image_height, int(y2 - height * 0.5))],
        
        # Upper/lower mirroring with significant vertical shift
        lambda: [max(0, int(x1 - width * 0.5)), max(0, int(image_height - y2 - height)), 
                 min(image_width, int(x2 - width * 0.5)), min(image_height, int(image_height - y1 + height))],
                 
        # Completely different regions (quadrant shifts to opposite sides)
        lambda: [max(0, int(image_width * 0.05)), max(0, int(image_height * 0.05)), 
                 min(image_width, int(image_width * 0.2)), min(image_height, int(image_height * 0.2))],
        lambda: [max(0, int(image_width * 0.8)), max(0, int(image_height * 0.8)), 
                 min(image_width, int(image_width * 0.95)), min(image_height, int(image_height * 0.95))],
                 
        # Vertical flip only
        lambda: [x1, max(0, int(image_height - y2)), 
                 x2, min(image_height, int(image_height - y1))],
                 
        # Diagonal displacement (move to opposite quadrant)
        lambda: [max(0, int(image_width - x2 - width)), max(0, int(image_height - y2 - height)), 
                 min(image_width, int(image_width - x1 + width)), min(image_height, int(image_height - y1 + height))],
                 
        # Very thin box with same center
        lambda: [max(0, int(center_x - width * 0.1)), max(0, int(center_y - height * 2.0)), 
                 min(image_width, int(center_x + width * 0.1)), min(image_height, int(center_y + height * 2.0))],
                 
        # Very wide box with same center
        lambda: [max(0, int(center_x - width * 2.0)), max(0, int(center_y - height * 0.1)), 
                 min(image_width, int(center_x + width * 2.0)), min(image_height, int(center_y + height * 0.1))],
    ]
    
    # If other bboxes are provided, add them as potential wrong answers
    # But only if they're significantly different from the correct bbox
    if other_bboxes:
        for bbox in other_bboxes[:3]:  # Limit to first 3 to avoid too many similar options
            if bbox != correct_bbox:
                # Calculate IoU (Intersection over Union) to ensure boxes are different enough
                x1_i = max(correct_bbox[0], bbox[0])
                y1_i = max(correct_bbox[1], bbox[1])
                x2_i = min(correct_bbox[2], bbox[2])
                y2_i = min(correct_bbox[3], bbox[3])
                
                if x2_i <= x1_i or y2_i <= y1_i:  # No overlap
                    wrong_bbox_strategies.append(lambda b=bbox: b)
                else:
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    area1 = (correct_bbox[2] - correct_bbox[0]) * (correct_bbox[3] - correct_bbox[1])
                    area2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union
                    
                    # Only include if IoU is small (boxes are very different)
                    if iou < 0.3:
                        wrong_bbox_strategies.append(lambda b=bbox: b)
    
    max_attempts = 15  # 增加尝试次数以确保找到不重复的bbox
    for _ in range(max_attempts):
        wrong_bbox = random.choice(wrong_bbox_strategies)()
        
        # 确保生成的bbox与已有的错误bbox不重复
        is_duplicate = False
        for existing_bbox in existing_wrong_bboxes:
            if wrong_bbox == existing_bbox:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
            
        # Check if the wrong box is different enough from the correct box
        x1_i = max(correct_bbox[0], wrong_bbox[0])
        y1_i = max(correct_bbox[1], wrong_bbox[1])
        x2_i = min(correct_bbox[2], wrong_bbox[2])
        y2_i = min(correct_bbox[3], wrong_bbox[3])
        
        # Calculate IoU
        if x2_i <= x1_i or y2_i <= y1_i:  # No overlap
            return wrong_bbox  # Perfect, no overlap at all
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (correct_bbox[2] - correct_bbox[0]) * (correct_bbox[3] - correct_bbox[1])
        area2 = (wrong_bbox[2] - wrong_bbox[0]) * (wrong_bbox[3] - wrong_bbox[1])
        union = area1 + area2 - intersection
        iou = intersection / union
        
        if iou < 0.3:  # Low overlap, boxes are different enough
            return wrong_bbox
    
    # 如果所有策略都失败，生成一个确保不重复的极端不同选项
    extreme_options = [
        [max(0, int(image_width * 0.05)), max(0, int(image_height * 0.05)), 
         min(image_width, int(image_width * 0.2)), min(image_height, int(image_height * 0.2))],
        [max(0, int(image_width * 0.8)), max(0, int(image_height * 0.8)), 
         min(image_width, int(image_width * 0.95)), min(image_height, int(image_height * 0.95))],
        [max(0, int(image_width - x2 - width)), max(0, int(image_height - y2 - height)), 
         min(image_width, int(image_width - x1 + width)), min(image_height, int(image_height - y1 + height))]
    ]
    
    # 尝试极端选项中找到一个不重复的
    for option in extreme_options:
        if option not in existing_wrong_bboxes:
            return option
            
    # 如果所有尝试都失败，生成一个随机但不同的bbox
    while True:
        random_box = [
            random.randint(0, max(1, image_width - 100)),
            random.randint(0, max(1, image_height - 100)),
            random.randint(min(image_width - 50, 100), image_width),
            random.randint(min(image_height - 50, 100), image_height)
        ]
        
        if random_box not in existing_wrong_bboxes and random_box != correct_bbox:
            return random_box

def generate_wrong_count(correct_count: int, max_possible: int = 32) -> List[int]:
    """Generate wrong count numbers with increased differences for multiple choice options"""
    wrong_counts = set()
    
    # Strategy 1: correct count with larger differences (±3-5)
    candidates = [max(0, correct_count - 5), max(0, correct_count - 3), 
                  correct_count + 3, correct_count + 5]
    
    # Strategy 2: correct count with smaller differences for more difficult cases
    candidates.extend([max(0, correct_count - 1), max(0, correct_count - 2), 
                       correct_count + 1, correct_count + 2])
    
    # Strategy 3: drastically different values
    candidates.extend([0, max_possible, max(0, correct_count // 2), correct_count * 2])
    
    # Filter out the correct answer and select 3 wrong answers
    candidates = [c for c in candidates if c != correct_count and 0 <= c <= max_possible]
    wrong_counts = set(random.sample(candidates, min(3, len(candidates))))
    
    # If we couldn't get 3 wrong answers, pad with more random numbers
    while len(wrong_counts) < 3:
        new_wrong = random.randint(0, max_possible)
        if new_wrong != correct_count and new_wrong not in wrong_counts:
            wrong_counts.add(new_wrong)
    
    return list(wrong_counts)

def format_answer_with_tooth_id(answer: str) -> str:
    """Format answer to use appropriate format based on content"""
    # 修改：如果答案中包含wisdom tooth，直接使用"wisdom tooth #ID"格式，不要加"on tooth"
    if "#" in answer and "wisdom tooth" in answer.lower():
        return answer
    
    # 处理其他普通牙齿的答案格式
    if "#" in answer:
        # Extract the tooth ID
        match = re.search(r'(.*)\s+#(\d+)$', answer)
        if match:
            condition = match.group(1).strip()
            tooth_id = match.group(2)
            return f"{condition} on tooth #{tooth_id}"
    return answer

def generate_multiple_choice_options(correct_answer: str, wrong_options: List[str], 
                                    shuffle: bool = True) -> Tuple[Dict[str, str], str]:
    """Generate ABCD options with correct answer and wrong options"""
    # Format the correct answer to use appropriate format
    formatted_correct = format_answer_with_tooth_id(correct_answer)
    
    # Format wrong options to use appropriate format
    formatted_wrong = [format_answer_with_tooth_id(opt) for opt in wrong_options[:3]]
    
    all_options = [formatted_correct] + formatted_wrong  # Limit to 3 wrong options
    
    if shuffle:
        random.shuffle(all_options)
    
    correct_index = all_options.index(formatted_correct)
    correct_letter = chr(65 + correct_index)  # A, B, C, or D
    
    options = {
        "A": all_options[0],
        "B": all_options[1],
        "C": all_options[2],
        "D": all_options[3] if len(all_options) > 3 else "None of the above"
    }
    
    return options, correct_letter

def ensure_category_keywords(question: str, category: str) -> str:
    """Ensure the question contains keywords related to its category"""
    keywords = CATEGORY_KEYWORDS.get(category, [category.split()[0].lower()])
    if not any(keyword.lower() in question.lower() for keyword in keywords):
        return f"{question}"
    return question

def format_question_with_bbox(template: str, bbox: List[int]) -> str:
    """Format a question template with a bounding box"""
    if "{box_2d}" in template:
        return template.replace("{box_2d}", str(bbox))
    return template

def format_question_with_tooth_id(template: str, tooth_id: Union[str, int]) -> str:
    """Format a question template with a tooth ID"""
    if "{tooth_id}" in template:
        return template.replace("{tooth_id}", str(tooth_id))
    return template

def get_valid_tooth_id(exclude_id: str = None) -> str:
    """生成有效的牙齿ID，符合牙科编号系统(11-18, 21-28, 31-38, 41-48)"""
    # 定义所有有效的牙齿ID区间
    valid_ranges = [
        range(11, 19),  # 右上象限
        range(21, 29),  # 左上象限
        range(31, 39),  # 左下象限
        range(41, 49)   # 右下象限
    ]
    
    # 将所有有效ID合并到一个列表中
    all_valid_ids = [str(tooth_id) for r in valid_ranges for tooth_id in r]
    
    # 如果需要排除特定ID，将其从列表中移除
    if exclude_id and exclude_id in all_valid_ids:
        all_valid_ids.remove(exclude_id)
    
    # 随机选择一个有效ID
    return random.choice(all_valid_ids)

def format_question_with_point_2d(template: str, point_2d: List[int]) -> str:
    """Format a question template with a 2D point"""
    if "{point_2d[0]}" in template and "{point_2d[1]}" in template:
        return template.replace("{point_2d[0]}", str(point_2d[0])).replace("{point_2d[1]}", str(point_2d[1]))
    return template

def generate_box_identification_question(template: str, box_2d: List[int], 
                                         target_type: str, target_id: Optional[str] = None) -> Dict:
    """Generate a question asking what is in a specific bounding box"""
    question = format_question_with_bbox(template, box_2d)
    
    # 修改问题文本，对于历史治疗类型，使用"historical treatment"来提问
    if "Dental filling" in target_type or "Dental crown" in target_type or "Root canal treatment" in target_type or "Dental implant" in target_type:
        # 替换问题中的"object"为"historical treatment"
        question = question.replace("What object is", "What historical treatment is")
        question = question.replace("any object", "any historical dental treatment")
        question = question.replace("what object is", "what historical treatment is")
    
    # Define the correct answer based on target type and ID
    if "Wisdom tooth" in target_type and target_id and target_id.lower() != "unknown":
        # 修改：对于智齿，直接使用"wisdom tooth #ID"格式
        correct_answer = f"Wisdom tooth #{target_id}"
    elif target_id and target_id.lower() != "unknown":
        correct_answer = f"{target_type} #{target_id}"
    else:
        correct_answer = target_type
    
    # Generate wrong options
    wrong_types = ["Normal tooth", "Nothing detected", "Unclear region"]
    
    # 根据目标类型生成更相关的错误选项
    if target_type == "Dental caries":
        wrong_types = ["Periapical lesion", "Normal tooth", "Dental filling"]
    elif target_type == "Periapical lesion":
        wrong_types = ["Dental caries", "Normal tooth", "Bone loss"]
    elif target_type == "Dental filling":
        wrong_types = ["Dental crown", "Normal tooth", "Root canal treatment"]
    elif target_type == "Dental crown":
        wrong_types = ["Dental filling", "Normal tooth", "Dental implant"]
    elif target_type == "Root canal treatment":
        wrong_types = ["Dental filling", "Normal tooth", "Periapical lesion"]
    elif target_type == "Dental implant":
        wrong_types = ["Dental crown", "Normal tooth", "Bone loss"]
    elif target_type == "Wisdom tooth":
        wrong_types = ["Normal tooth", "Impacted tooth", "Missing tooth"]
    elif target_type == "Impacted tooth":
        wrong_types = ["Normal tooth", "Wisdom tooth", "Missing tooth"]
    elif target_type == "Missing tooth":
        wrong_types = ["Normal tooth", "Impacted tooth", "Dental implant"]
    elif target_type == "Mandibular canal":
        wrong_types = ["Maxillary sinus", "Bone structure", "Nerve pathway"]
    elif target_type == "Maxillary sinus":
        wrong_types = ["Mandibular canal", "Bone structure", "Nasal cavity"]
    
    # If target_id is provided, add valid tooth numbers to the wrong options
    if target_id and target_id.lower() != "unknown":
        wrong_ids = []
        for _ in range(min(2, len(wrong_types))):
            # 使用合法的牙齿ID
            wrong_id = get_valid_tooth_id(target_id)
            while wrong_id in wrong_ids:
                wrong_id = get_valid_tooth_id(target_id)
            wrong_ids.append(wrong_id)
        
        wrong_options = []
        for i, wrong_id in enumerate(wrong_ids):
            if "Wisdom tooth" in wrong_types[i].lower():
                # 修改：对于智齿错误选项，也使用直接格式
                wrong_options.append(f"Wisdom tooth #{wrong_id}")
            else:
                wrong_options.append(f"{wrong_types[i]} #{wrong_id}")
        
        # Add at least one option with a different type but same tooth ID
        if len(wrong_types) > 2:
            if "wisdom" in wrong_types[2].lower():
                wrong_options.append(f"Wisdom tooth #{target_id}")
            else:
                wrong_options.append(f"{wrong_types[2]} #{target_id}")
    else:
        wrong_options = wrong_types
    
    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
    
    # Use appropriate format in explanation based on target type
    if "Wisdom tooth" in target_type and target_id and target_id.lower() != "unknown":
        explanation = f"The bounding box {box_2d} contains a wisdom tooth #{target_id}."
    elif target_id and target_id.lower() != "unknown":
        explanation = f"The bounding box {box_2d} contains {target_type} on tooth #{target_id}."
    else:
        explanation = f"The bounding box {box_2d} contains {target_type}."
    
    return {
        "question": question,
        "options": options,
        "answer": correct_letter,
        "explanation": explanation
    }

def generate_regular_teeth_spatial_questions(parsed_data: Dict, num_questions: int = 2) -> List[Dict]:
    """
    Generate spatial questions for exactly num_questions random regular teeth,
    focusing on point_2d coordinates rather than box_2d.
    
    Parameters:
        parsed_data: The parsed X-ray data
        num_questions: Number of questions to generate (default: 2)
    
    Returns:
        List of generated question dictionaries
    """
    if "Teeth visibility with center points" not in parsed_data:
        return []
    
    qa_pairs = []
    teeth_data = parsed_data["Teeth visibility with center points"]
    templates = TEMPLATE_DICT["Teeth visibility with center points"]["spatial"]
    
    # Filter out teeth with "unknown" tooth_id and ensure they have point_2d
    valid_teeth = [
        item for item in teeth_data
        if "tooth_id" in item 
        and item["tooth_id"] 
        and item["tooth_id"].lower() != "unknown"
        and "point_2d" in item
    ]
    
    if not valid_teeth:
        return []
    
    # If we have fewer valid teeth than requested questions, use all available
    if len(valid_teeth) <= num_questions:
        selected_teeth = valid_teeth
    else:
        # Randomly select exactly num_questions different teeth
        selected_teeth = random.sample(valid_teeth, num_questions)
    
    # Generate questions for each selected tooth
    for tooth_item in selected_teeth:
        tooth_id = tooth_item["tooth_id"]
        point_2d = tooth_item["point_2d"]
        
        # Randomly decide whether to ask:
        # 1. Given tooth_id, ask for coordinates (50% chance)
        # 2. Given coordinates, ask for tooth_id (50% chance)
        if random.random() < 0.5:
            # Type 1: Ask for coordinates given tooth_id
            # Select a template asking for coordinates of a specific tooth
            id_to_coord_templates = [t for t in templates if "{tooth_id}" in t and "{point_2d[0]}" not in t]
            if not id_to_coord_templates:
                # Fallback to any template with tooth_id
                id_to_coord_templates = [t for t in templates if "{tooth_id}" in t]
                if not id_to_coord_templates:
                    continue
                    
            template = random.choice(id_to_coord_templates)
            question = format_question_with_tooth_id(template, tooth_id)
            
            # The correct answer is the point_2d coordinates
            correct_answer = str(point_2d)
            
            # Generate incorrect point_2d options
            other_points = [item["point_2d"] for item in teeth_data if item != tooth_item and "point_2d" in item]
            wrong_points = []
            
            # Select 3 different points for wrong options
            if other_points:
                # If we have other points, select from them
                if len(other_points) >= 3:
                    wrong_points = random.sample(other_points, 3)
                else:
                    # If we have fewer than 3 other points, use what we have and generate the rest
                    wrong_points = other_points.copy()
                    while len(wrong_points) < 3:
                        # Generate a random point in the vicinity of the image
                        random_x = random.randint(max(0, point_2d[0] - 500), min(1920, point_2d[0] + 500))
                        random_y = random.randint(max(0, point_2d[1] - 500), min(1080, point_2d[1] + 500))
                        wrong_point = [random_x, random_y]
                        if wrong_point not in wrong_points and wrong_point != point_2d:
                            wrong_points.append(wrong_point)
            else:
                # Generate 3 random points
                for _ in range(3):
                    random_x = random.randint(max(0, point_2d[0] - 500), min(1920, point_2d[0] + 500))
                    random_y = random.randint(max(0, point_2d[1] - 500), min(1080, point_2d[1] + 500))
                    wrong_point = [random_x, random_y]
                    wrong_points.append(wrong_point)
            
            # Create multiple choice options
            options, correct_letter = generate_multiple_choice_options(
                correct_answer, 
                [str(point) for point in wrong_points]
            )
            
            # Create question dict
            qa = {
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"Tooth #{tooth_id} is located at coordinates {point_2d}."
            }
            
        else:
            # Type 2: Ask for tooth_id given coordinates
            # Select a template asking for tooth ID at specific coordinates
            coord_to_id_templates = [t for t in templates if "{point_2d[0]}" in t and "{point_2d[1]}" in t]
            if not coord_to_id_templates:
                # Fallback to asking for tooth ID given box
                coord_to_id_templates = [t for t in templates if "{box_2d}" in t]
                if coord_to_id_templates:
                    # Adapt the template to use point_2d instead of box_2d
                    template = random.choice(coord_to_id_templates)
                    question = template.replace("{box_2d}", str(point_2d))
                else:
                    continue
            else:
                template = random.choice(coord_to_id_templates)
                question = format_question_with_point_2d(template, point_2d)
            
            # The correct answer is the tooth_id
            correct_answer = str(tooth_id)
            
            # Generate incorrect tooth_id options
            other_ids = [item["tooth_id"] for item in teeth_data if item != tooth_item and "tooth_id" in item]
            wrong_ids = []
            
            # Select 3 different tooth IDs for wrong options
            if other_ids:
                if len(other_ids) >= 3:
                    wrong_ids = random.sample(other_ids, 3)
                else:
                    # If we have fewer than 3 other IDs, use what we have and generate the rest
                    wrong_ids = other_ids.copy()
                    valid_ranges = [range(11, 19), range(21, 29), range(31, 39), range(41, 49)]
                    all_valid_ids = [str(id) for r in valid_ranges for id in r]
                    
                    # Filter out IDs we're already using
                    remaining_ids = [id for id in all_valid_ids if id not in wrong_ids and id != tooth_id]
                    
                    # Add random IDs until we have 3
                    while len(wrong_ids) < 3 and remaining_ids:
                        random_id = random.choice(remaining_ids)
                        wrong_ids.append(random_id)
                        remaining_ids.remove(random_id)
            else:
                # Generate 3 random tooth IDs
                valid_ranges = [range(11, 19), range(21, 29), range(31, 39), range(41, 49)]
                all_valid_ids = [str(id) for r in valid_ranges for id in r]
                
                # Filter out the correct tooth_id
                remaining_ids = [id for id in all_valid_ids if id != tooth_id]
                
                # Select 3 random IDs
                if len(remaining_ids) >= 3:
                    wrong_ids = random.sample(remaining_ids, 3)
                else:
                    # Unlikely, but just in case
                    wrong_ids = remaining_ids
            
            # Create multiple choice options
            options, correct_letter = generate_multiple_choice_options(
                correct_answer, 
                wrong_ids
            )
            
            # Create question dict
            qa = {
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"The tooth at coordinates {point_2d} is tooth #{tooth_id}."
            }
        
        qa["category"] = "Teeth visibility with center points"
        qa["question"] = ensure_category_keywords(qa["question"], "Teeth visibility with center points")
        qa_pairs.append(qa)
    
    return qa_pairs

def generate_teeth_count_question(parsed_data: Dict) -> Optional[Dict]:
    """Generate exactly one count question for visible teeth"""
    if "Teeth visibility with center points" not in parsed_data:
        return None
    
    teeth_data = parsed_data["Teeth visibility with center points"]
    if not teeth_data:
        return None
    
    # 选择一个计数问题模板
    count_templates = TEMPLATE_DICT["Teeth visibility with center points"]["count"]
    if not count_templates:
        return None
    
    template = random.choice(count_templates)
    
    # 计算可见牙齿的数量
    count = len(teeth_data)
    wrong_counts = generate_wrong_count(count)
    
    # 创建多项选择题
    options, correct_letter = generate_multiple_choice_options(
        str(count), 
        [str(c) for c in wrong_counts]
    )
    
    # 创建问题字典
    qa = {
        "question": template,
        "options": options,
        "answer": correct_letter,
        "explanation": f"There are {count} teeth visible in the panoramic X-ray."
    }
    
    qa["category"] = "Teeth visibility with center points"
    qa["question"] = ensure_category_keywords(qa["question"], "Teeth visibility with center points")
    
    return qa

def generate_tooth_based_treatment_question(item: Dict, templates: Dict) -> Optional[Dict]:
    """生成基于牙齿ID的历史治疗问题，问'这颗牙齿上有什么治疗'，而不是'这个坐标有什么'"""
    if "tooth_id" not in item or not item["tooth_id"] or item["tooth_id"].lower() == "unknown":
        return None
    
    tooth_id = item["tooth_id"]
    
    # 确定具体的治疗类型 (filling, crown, root canal treatment, implant)
    treatment_type = None
    if "treatment_type" in item:
        treatment_type = item["treatment_type"].lower()
    elif "label" in item:
        treatment_type = item["label"].lower()
    else:
        return None  # 无法确定治疗类型
    
    # 映射到标准化的具体治疗类型名称
    specific_treatment = None
    if "filling" in treatment_type:
        specific_treatment = "Dental filling"
    elif "crown" in treatment_type:
        specific_treatment = "Dental crown"
    elif "root canal" in treatment_type:
        specific_treatment = "Root canal treatment"
    elif "implant" in treatment_type:
        specific_treatment = "Dental implant"
    else:
        specific_treatment = "Dental treatment"  # 默认值
    
    # 选择一个基于牙齿ID的治疗问题模板
    tooth_based_templates = templates.get("tooth_based", [])
    if not tooth_based_templates:
        return None
    
    template = random.choice(tooth_based_templates)
    question = format_question_with_tooth_id(template, tooth_id)
    
    # 正确答案为具体治疗类型
    correct_answer = specific_treatment
    
    # 生成错误选项 - 使用其他治疗类型
    wrong_treatments = ["Dental filling", "Dental crown", "Root canal treatment", "Dental implant", "No treatment"]
    if specific_treatment in wrong_treatments:
        wrong_treatments.remove(specific_treatment)
    
    wrong_options = random.sample(wrong_treatments, 3)
    
    # 创建多项选择题
    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
    
    # 生成详细的解释
    explanation = f"Tooth #{tooth_id} has a {specific_treatment.lower()}."
    
    return {
        "question": question,
        "options": options,
        "answer": correct_letter,
        "explanation": explanation
    }

def generate_periapical_lesion_type_question(item: Dict, templates: Dict) -> Optional[Dict]:
    """生成询问根尖周炎类型的问题"""
    if "tooth_id" not in item or not item["tooth_id"] or item["tooth_id"].lower() == "unknown":
        return None
    
    # 检查是否有明确的根尖周炎类型
    lesion_type = None
    if "lesion_type" in item:
        lesion_type = item["lesion_type"]
    else:
        return None  # 无法确定根尖周炎类型
    
    tooth_id = item["tooth_id"]
    
    # 选择一个询问根尖周炎类型的模板
    type_templates = templates.get("type", [])
    if not type_templates:
        return None
    
    template = random.choice(type_templates)
    question = format_question_with_tooth_id(template, tooth_id)
    
    # 正确答案是特定的根尖周炎类型
    correct_answer = lesion_type
    
    # 生成错误选项 - 使用其他可能的根尖周炎类型
    wrong_types = ["Periapical abscess", "Periapical cyst", "Periapical granuloma", "Periapical sclerosis", 
                  "Periapical rarefaction", "Condensing osteitis"]
    
    # 移除正确答案
    if correct_answer in wrong_types:
        wrong_types.remove(correct_answer)
    
    wrong_options = random.sample(wrong_types, 3)
    
    # 创建多项选择题
    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
    
    # 生成详细的解释
    explanation = f"The periapical lesion on tooth #{tooth_id} is a {correct_answer.lower()}."
    
    return {
        "question": question,
        "options": options,
        "answer": correct_letter,
        "explanation": explanation
    }

def generate_question_for_historical_treatment(item: Dict, templates: Dict) -> Optional[Dict]:
    """为历史治疗bbox生成问题，统一使用historical treatment术语，但在答案中保留子类别"""
    if "box_2d" not in item:
        return None
    
    box_2d = item["box_2d"]
    
    # 确定具体的治疗类型 (filling, crown, root canal treatment, implant)
    treatment_type = None
    if "treatment_type" in item:
        treatment_type = item["treatment_type"].lower()
    elif "label" in item:
        treatment_type = item["label"].lower()
    else:
        return None  # 无法确定治疗类型
    
    # 映射到标准化的具体治疗类型名称
    specific_treatment = None
    if "filling" in treatment_type:
        specific_treatment = "Dental filling"
    elif "crown" in treatment_type:
        specific_treatment = "Dental crown"
    elif "root canal" in treatment_type:
        specific_treatment = "Root canal treatment"
    elif "implant" in treatment_type:
        specific_treatment = "Dental implant"
    else:
        specific_treatment = "Dental treatment"  # 默认值
    
    tooth_id = item.get("tooth_id")
    
    # Skip if tooth_id is "unknown"
    if tooth_id and tooth_id.lower() == "unknown":
        return None
    
    # 选择一个询问bbox内容的模板 - 确保使用"historical treatment"模板
    spatial_templates = templates.get("spatial", [])
    historical_templates = [t for t in spatial_templates if 
                          ("historical treatment" in t.lower() or 
                           "historical dental treatment" in t.lower())]
    
    if not historical_templates:
        # 如果没有找到专门的historical treatment模板，使用常规bbox模板
        bbox_templates = [t for t in spatial_templates if "{box_2d}" in t]
        if not bbox_templates:
            return None
        template = random.choice(bbox_templates)
    else:
        template = random.choice(historical_templates)
    
    question = format_question_with_bbox(template, box_2d)
    
    # 确保问题中包含"historical treatment"术语
    if "historical treatment" not in question.lower() and "historical dental treatment" not in question.lower():
        question = question.replace("What object is", "What historical treatment is")
        question = question.replace("any object", "any historical dental treatment")
        question = question.replace("what object is", "what historical treatment is")
    
    # 生成答案选项，确保答案是具体的治疗类型
    if tooth_id:
        correct_answer = f"{specific_treatment} #{tooth_id}"
    else:
        correct_answer = specific_treatment
    
    # 生成错误选项 - 使用其他治疗类型
    wrong_treatments = ["Dental filling", "Dental crown", "Root canal treatment", "Dental implant"]
    wrong_treatments.remove(specific_treatment)
    
    # 生成带牙齿ID的错误选项
    wrong_options = []
    if tooth_id:
        # 至少一个选项保持相同治疗类型但牙齿ID不同
        wrong_id = get_valid_tooth_id(tooth_id)
        wrong_options.append(f"{specific_treatment} #{wrong_id}")
        
        # 其他选项使用不同治疗类型但相同牙齿ID
        wrong_options.append(f"{wrong_treatments[0]} #{tooth_id}")
        
        # 最后一个选项使用不同治疗类型和不同牙齿ID
        wrong_id2 = get_valid_tooth_id(tooth_id)
        while wrong_id2 == wrong_id:
            wrong_id2 = get_valid_tooth_id(tooth_id)
        wrong_options.append(f"{wrong_treatments[1]} #{wrong_id2}")
    else:
        wrong_options = wrong_treatments[:3]
    
    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
    
    # 生成详细的解释，明确提及具体治疗类型，使用'on tooth #ID'格式
    if tooth_id:
        explanation = f"The bounding box {box_2d} contains a {specific_treatment.lower()} on tooth #{tooth_id}."
    else:
        explanation = f"The bounding box {box_2d} contains a {specific_treatment.lower()}."
    
    return {
        "question": question,
        "options": options,
        "answer": correct_letter,
        "explanation": explanation
    }

def generate_question_for_special_bbox(category: str, item: Dict, 
                                      templates: Dict, parsed_data: Dict) -> Optional[Dict]:
    """Generate a question for a special (non-regular teeth) bounding box"""
    if "box_2d" not in item:
        return None
    
    # Skip if tooth_id is "unknown"
    if "tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() == "unknown":
        return None
    
    box_2d = item["box_2d"]
    
    # Select a spatial template that asks about what's in a bounding box
    spatial_templates = templates.get("spatial", [])
    bbox_templates = [t for t in spatial_templates if "{box_2d}" in t]
    
    if not bbox_templates:
        return None
    
    template = random.choice(bbox_templates)
    question = format_question_with_bbox(template, box_2d)
    
    # Determine target type and ID based on category
    target_type = None
    target_id = None
    
    if category == "Dental caries detection":
        target_type = "Dental caries"
        target_id = item.get("tooth_id")
    elif category == "Periapical lesions detection":
        target_type = "Periapical lesion"
        target_id = item.get("tooth_id")
    # 修改治疗类型判断逻辑，不再使用通用的"Historical treatments"
    elif category == "Dental fillings":
        target_type = "Dental filling"
        target_id = item.get("tooth_id")
    elif category == "Dental crowns":
        target_type = "Dental crown"
        target_id = item.get("tooth_id")
    elif category == "Root canal treatments":
        target_type = "Root canal treatment"
        target_id = item.get("tooth_id")
    elif category == "Dental implants":
        target_type = "Dental implant"
        target_id = item.get("tooth_id")
    elif category == "Historical treatments":
        # 处理历史治疗 - 通过treatment_type或label确定具体类型
        treatment_type = item.get("treatment_type", "").lower()
        if not treatment_type and "label" in item:
            treatment_type = item["label"].lower()
        
        if "filling" in treatment_type:
            target_type = "Dental filling"
        elif "crown" in treatment_type:
            target_type = "Dental crown"
        elif "root canal" in treatment_type:
            target_type = "Root canal treatment"
        elif "implant" in treatment_type:
            target_type = "Dental implant"
        else:
            target_type = "Dental treatment"  # 默认值
        
        target_id = item.get("tooth_id")
    elif category == "Wisdom teeth detection":
        target_type = "Wisdom tooth"
        target_id = item.get("tooth_id")
    elif category == "Missing teeth detection":
        target_type = "Missing tooth"
        target_id = item.get("tooth_id")
    elif category == "Non-wisdom impacted teeth detection":
        target_type = "Impacted tooth"
        target_id = item.get("tooth_id")
    elif category == "Bone loss detection":
        target_type = "Bone loss"
    elif category == "Mandibular canal visibility":
        target_type = "Mandibular canal"
    elif category == "Maxillary sinuses visibility":
        target_type = "Maxillary sinus"
    
    if not target_type:
        return None
    
    return generate_box_identification_question(question, box_2d, target_type, target_id)

def generate_questions_for_category(parsed_data: Dict, category: str, 
                                   existing_questions: Optional[Set[str]] = None,
                                   question_types_used: Optional[Dict[str, List[str]]] = None,
                                   used_items: Optional[Set[str]] = None) -> List[Dict]:
    """Generate all possible questions for a category with special focus on bounding boxes"""
    if existing_questions is None:
        existing_questions = set()
    
    if question_types_used is None:
        question_types_used = {}
        
    if used_items is None:
        used_items = set()  # Track used bbox+tooth_id combinations
    
    if category not in parsed_data or not parsed_data[category]:
        return []
    
    if category not in TEMPLATE_DICT:
        return []
    
    qa_pairs = []
    category_data = parsed_data[category]
    templates = TEMPLATE_DICT[category]
    
    # 确保每个类别的question_types_used有记录
    if category not in question_types_used:
        question_types_used[category] = []
    
    # Process special spatial templates first (ensure all special bboxes are used)
    if category in SPECIAL_CONDITIONS:
        # Filter out items with unknown tooth_id
        filtered_items = [
            item for item in category_data
            if not ("tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() == "unknown")
        ]
        
        for item in filtered_items:
            if "box_2d" in item:
                # Create a unique key for this box/tooth combination
                item_key = str(item.get("box_2d", "")) + "|" + str(item.get("tooth_id", ""))
                
                # Skip if this item has already been used for a question
                if item_key in used_items:
                    continue
                    
                qa = generate_question_for_special_bbox(category, item, templates, parsed_data)
                if qa:
                    qa["category"] = category
                    qa["question"] = ensure_category_keywords(qa["question"], category)
                    qa_pairs.append(qa)
                    used_items.add(item_key)
    
    # 特殊处理根尖周炎类型问题
    if category == "Periapical lesions detection":
        # 为每个有lesion_type的根尖周炎项生成类型问题
        valid_items = [
            item for item in category_data
            if "tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() != "unknown"
            and "lesion_type" in item and item["lesion_type"]
        ]
        
        for item in valid_items:
            # 确保该牙齿的这个问题方向还没被问过
            tooth_id = item["tooth_id"]
            item_key = f"lesion_type|{tooth_id}"
            if item_key in used_items:
                continue
                
            qa = generate_periapical_lesion_type_question(item, templates)
            if qa:
                qa["category"] = category
                qa["question"] = ensure_category_keywords(qa["question"], category)
                qa_pairs.append(qa)
                used_items.add(item_key)
    
    # 特殊处理历史治疗类的tooth_based问题
    if category == "Historical treatments" and "tooth_based" in templates:
        # 为每个有效的历史治疗项生成基于牙齿ID的问题
        valid_items = [
            item for item in category_data
            if "tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() != "unknown"
        ]
        
        # 按牙齿ID分组
        teeth_with_treatments = {}
        for item in valid_items:
            tooth_id = item["tooth_id"]
            if tooth_id not in teeth_with_treatments:
                teeth_with_treatments[tooth_id] = []
            teeth_with_treatments[tooth_id].append(item)
        
        # 为每颗牙齿生成一个问题
        for tooth_id, items in teeth_with_treatments.items():
            # 确保该牙齿的这个问题方向还没被问过
            item_key = f"tooth_based|{tooth_id}"
            if item_key in used_items:
                continue
                
            # 随机选择一个该牙齿的治疗项生成问题
            item = random.choice(items)
            qa = generate_tooth_based_treatment_question(item, templates)
            if qa:
                qa["category"] = category
                qa["question"] = ensure_category_keywords(qa["question"], category)
                qa_pairs.append(qa)
                used_items.add(item_key)
    
    # Process count templates if available - 限制每个类别只选择一个count问题 (但特殊处理Teeth visibility)
    if "count" in templates and category_data and "count" not in question_types_used[category] and category != "Teeth visibility with center points":
        # 随机选择一个count模板
        template = random.choice(templates["count"])
        if template not in existing_questions:
            count = len(category_data)
            wrong_counts = generate_wrong_count(count)
            options, correct_letter = generate_multiple_choice_options(str(count), [str(c) for c in wrong_counts])
            
            # 生成更明确的友好类别名称
            friendly_category = category.lower()
            friendly_category = friendly_category.replace('detection', '').replace('visibility', '').strip()
            
            qa = {
                "question": template,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {count} {friendly_category} in the panoramic X-ray."
            }
            
            qa["category"] = category
            qa["question"] = ensure_category_keywords(qa["question"], category)
            qa_pairs.append(qa)
            existing_questions.add(template)
            question_types_used[category].append("count")
    
    # Process spatial templates for this category (except for regular teeth)
    if "spatial" in templates and category != "Teeth visibility with center points":
        # Process non-box_2d templates first
        spatial_templates = templates["spatial"]
        non_bbox_templates = [t for t in spatial_templates if "{box_2d}" not in t]
        
        for template in non_bbox_templates:
            if template in existing_questions:
                continue
                
            # Get remaining items that haven't been used for questions yet
            remaining_items = [
                item for item in category_data 
                if str(item.get("box_2d", "")) + "|" + str(item.get("tooth_id", "")) not in used_items
                and not ("tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() == "unknown")
            ]
            
            if not remaining_items:
                continue
                
            item = random.choice(remaining_items)
            
            # Handle tooth_id templates
            if "{tooth_id}" in template and "tooth_id" in item and item["tooth_id"]:
                tooth_id = item["tooth_id"]
                
                question = format_question_with_tooth_id(template, tooth_id)
                
                if "box_2d" in item:
                    correct_bbox = item["box_2d"]
                    correct_bbox_str = str(correct_bbox)
                    
                    # Get other bboxes from the same category for more relevant wrong options
                    other_bboxes = [
                        other_item["box_2d"] for other_item in category_data 
                        if "box_2d" in other_item and other_item != item
                    ]
                    
                    # 改进的错误生成逻辑，确保错误选项不重复
                    wrong_bboxes = []
                    for _ in range(3):
                        wrong_bbox = generate_wrong_bbox(correct_bbox, other_bboxes=other_bboxes, 
                                                        existing_wrong_bboxes=wrong_bboxes)
                        wrong_bboxes.append(wrong_bbox)
                    
                    wrong_bbox_strs = [str(bbox) for bbox in wrong_bboxes]
                    
                    options, correct_letter = generate_multiple_choice_options(correct_bbox_str, wrong_bbox_strs)
                    
                    # 使用适当的格式生成解释
                    if category == "Dental caries detection":
                        explanation = f"The dental caries on tooth #{tooth_id} is located at the bounding box {correct_bbox_str}."
                    elif category == "Periapical lesions detection":
                        explanation = f"The periapical lesion associated with tooth #{tooth_id} is located at the bounding box {correct_bbox_str}."
                    elif category == "Wisdom teeth detection":
                        # 对于智齿使用特殊格式
                        explanation = f"Wisdom tooth #{tooth_id} is located at the bounding box {correct_bbox_str}."
                    elif category == "Missing teeth detection":
                        explanation = f"The missing tooth position #{tooth_id} is located at the bounding box {correct_bbox_str}."
                    elif category == "Non-wisdom impacted teeth detection":
                        explanation = f"The impacted tooth #{tooth_id} is located at the bounding box {correct_bbox_str}."
                    else:
                        explanation = f"Tooth #{tooth_id} is located at the bounding box {correct_bbox_str}."
                    
                    qa = {
                        "question": question,
                        "options": options,
                        "answer": correct_letter,
                        "explanation": explanation
                    }
                    
                    qa["category"] = category
                    qa["question"] = ensure_category_keywords(qa["question"], category)
                    qa_pairs.append(qa)
                    existing_questions.add(template)
                    
                    item_key = str(item.get("box_2d", "")) + "|" + str(item.get("tooth_id", ""))
                    used_items.add(item_key)
            
            # Handle point_2d templates
            elif all(p in template for p in ["{point_2d[0]}", "{point_2d[1]}"]) and "point_2d" in item:
                question = format_question_with_point_2d(template, item["point_2d"])
                
                if "tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() != "unknown":
                    tooth_id = item["tooth_id"]
                    
                    correct_answer = str(tooth_id)
                    
                    # Generate wrong options with other tooth IDs
                    all_teeth = []
                    if "Teeth visibility with center points" in parsed_data:
                        all_teeth = [
                            t.get("tooth_id") for t in parsed_data["Teeth visibility with center points"]
                            if "tooth_id" in t 
                            and t["tooth_id"] 
                            and t["tooth_id"].lower() != "unknown" 
                            and t["tooth_id"] != tooth_id
                        ]
                    
                    wrong_options = []
                    if all_teeth:
                        wrong_options = [str(random.choice(all_teeth)) for _ in range(3)]
                    else:
                        # If no other teeth available, generate random tooth IDs
                        tooth_id_int = int(tooth_id) if tooth_id.isdigit() else 0
                        wrong_options = [str(random.randint(11, 48)) for _ in range(3)]
                        wrong_options = [opt for opt in wrong_options if opt != str(tooth_id_int)]
                    
                    options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                    
                    qa = {
                        "question": question,
                        "options": options,
                        "answer": correct_letter,
                        "explanation": f"The tooth at coordinates {item['point_2d']} is tooth #{tooth_id}."
                    }
                    
                    qa["category"] = category
                    qa["question"] = ensure_category_keywords(qa["question"], category)
                    qa_pairs.append(qa)
                    existing_questions.add(template)
                    
                    item_key = str(item.get("point_2d", "")) + "|" + str(item.get("tooth_id", ""))
                    used_items.add(item_key)
    
    # 确保问题多样性 - 对于每个类别，生成所有可能的问题类型
    if len(qa_pairs) == 0 and category_data and "count" not in question_types_used[category] and category != "Teeth visibility with center points":
        # 尝试生成一个默认问题
        if "count" in templates:
            template = random.choice(templates["count"])
            count = len(category_data)
            wrong_counts = generate_wrong_count(count)
            options, correct_letter = generate_multiple_choice_options(str(count), [str(c) for c in wrong_counts])
            
            friendly_category = category.lower().replace('detection', '').replace('visibility', '').strip()
            
            qa = {
                "question": template,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {count} {friendly_category} in the panoramic X-ray."
            }
            
            qa["category"] = category
            qa["question"] = ensure_category_keywords(qa["question"], category)
            qa_pairs.append(qa)
            question_types_used[category].append("count")
    
    return qa_pairs

def process_historical_treatments(parsed_data: Dict, templates: Dict) -> Tuple[List[Dict], Set[str], Set[str]]:
    """
    特殊处理历史治疗问题，维持7:3比例 (基于牙齿ID:基于坐标)
    
    Args:
        parsed_data: 解析后的X射线数据
        templates: 历史治疗问题的模板
        
    Returns:
        tuple: (生成的问题列表, 已使用的项目集合, 已使用的牙齿问题类型集合)
    """
    treatment_questions = []
    used_items = set()
    used_tooth_types = set()
    
    # 只处理有效的历史治疗项目
    if "Historical treatments" not in parsed_data or not parsed_data["Historical treatments"]:
        return treatment_questions, used_items, used_tooth_types
    
    # 按牙齿ID分组所有治疗项目
    teeth_with_treatments = {}
    valid_treatment_items = []
    
    # 收集所有有效的治疗项目
    for item in parsed_data["Historical treatments"]:
        # 过滤掉无效或未知牙齿ID的项目
        if "tooth_id" in item and item["tooth_id"] and item["tooth_id"].lower() != "unknown":
            tooth_id = item["tooth_id"]
            if tooth_id not in teeth_with_treatments:
                teeth_with_treatments[tooth_id] = []
            teeth_with_treatments[tooth_id].append(item)
            valid_treatment_items.append(item)
    
    # 如果没有有效的治疗项目，则返回空结果
    if not valid_treatment_items:
        return treatment_questions, used_items, used_tooth_types
    
    # 计算可用的治疗项目总数
    total_treatments = len(teeth_with_treatments)  # 使用牙齿数量而非治疗项总数
    
    # 计算应生成的基于ID和基于坐标的问题数量
    # 确保至少有1个问题，并保持7:3的比例
    id_questions_count = max(1, int(total_treatments * 0.7))
    coord_questions_count = max(1, int(total_treatments * 0.3))
    
    # 确保问题总数不超过总牙齿数
    if id_questions_count + coord_questions_count > total_treatments:
        if total_treatments == 1:
            id_questions_count = 1
            coord_questions_count = 0
        else:
            # 调整为70:30的比例
            id_questions_count = int(total_treatments * 0.7)
            coord_questions_count = total_treatments - id_questions_count
    
    print(f"  Historical treatments: planning {id_questions_count} tooth-based and {coord_questions_count} coordinate-based questions")
    
    # 随机选择牙齿生成基于ID的问题
    if teeth_with_treatments:
        tooth_ids = list(teeth_with_treatments.keys())
        
        # 如果牙齿数量小于总需求数量，先生成所有可能的基于ID问题
        if len(tooth_ids) <= id_questions_count:
            selected_tooth_ids = tooth_ids
        else:
            # 否则随机选择指定数量的牙齿
            selected_tooth_ids = random.sample(tooth_ids, id_questions_count)
        
        # 生成基于ID的问题
        for tooth_id in selected_tooth_ids:
            item = random.choice(teeth_with_treatments[tooth_id])
            qa = generate_tooth_based_treatment_question(item, templates)
            if qa:
                qa["category"] = "Historical treatments"
                treatment_questions.append(qa)
                used_items.add(f"tooth_based|{tooth_id}")
                used_tooth_types.add(f"{tooth_id}_tooth_based")
                # 从可用牙齿列表中移除已使用的牙齿
                tooth_ids.remove(tooth_id)
    
    # 为剩余的牙齿生成基于坐标的问题
    # 优先选择未使用的牙齿
    remaining_teeth = tooth_ids  # 这些牙齿尚未生成基于ID的问题
    
    # 根据需要的基于坐标问题数量，选择牙齿
    selected_teeth_for_coord = []
    if remaining_teeth and coord_questions_count > 0:
        # 如果剩余牙齿数量小于需求数量，全部选择
        if len(remaining_teeth) <= coord_questions_count:
            selected_teeth_for_coord = remaining_teeth
        else:
            # 否则随机选择指定数量的牙齿
            selected_teeth_for_coord = random.sample(remaining_teeth, coord_questions_count)
    
    # 为选定的牙齿生成基于坐标的问题
    for tooth_id in selected_teeth_for_coord:
        items = teeth_with_treatments[tooth_id]
        if items:
            # 随机选择这颗牙齿的一个治疗项目
            item = random.choice(items)
            if "box_2d" in item:
                qa = generate_question_for_historical_treatment(item, templates)
                if qa:
                    qa["category"] = "Historical treatments"
                    treatment_questions.append(qa)
                    item_key = str(item.get("box_2d", "")) + "|" + str(tooth_id)
                    used_items.add(item_key)
                    used_tooth_types.add(f"{tooth_id}_coordinate")
    
    return treatment_questions, used_items, used_tooth_types

def process_json_files(input_folder: str, output_folder: str):
    """Process all JSON files in the input folder and add QA data without limiting question count"""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "loc_caption" not in data or "including:\n" not in data["loc_caption"]:
                print(f"Skipping {filename}: Invalid localization caption")
                continue
            
            loc_caption = data["loc_caption"].split('including:\n')[1].strip()
            parsed_data = parse_medical_string(loc_caption)

            # Identify all available categories, prioritizing special conditions
            available_categories = []
            special_available = []
            regular_available = []
            
            # 特殊处理历史治疗类别
            has_historical_treatments = False
            historical_treatment_questions = []
            
            # 跟踪每个牙齿ID和对应问题类型的使用情况
            used_tooth_question_types = set()
            
            qa_pairs = []
            used_templates = set()
            question_types_used = {}  # 跟踪每个类别中已经使用的问题类型 (count, spatial等)
            used_items = set()  # 跟踪已经使用过的bbox+tooth_id组合
            
            # 首先，处理历史治疗问题
            if "Historical treatments" in parsed_data and parsed_data["Historical treatments"]:
                has_historical_treatments = True
                
                # 跟踪已经使用过的bbox+tooth_id组合
                used_historical_items = set()
                
                # 处理历史治疗问题，保持7:3比例（基于牙齿ID:基于坐标）
                historical_treatment_questions, used_historical_items, used_tooth_question_types = process_historical_treatments(
                    parsed_data, 
                    TEMPLATE_DICT["Historical treatments"]
                )
                
                qa_pairs.extend(historical_treatment_questions)
                print(f"Generated {len(historical_treatment_questions)} historical treatment questions for {filename}")
            
            # 第二步，为每个JSON生成一个牙齿计数问题（必须生成）
            teeth_count_qa = generate_teeth_count_question(parsed_data)
            if teeth_count_qa:
                qa_pairs.append(teeth_count_qa)
                print(f"  - Added 1 count question for teeth visibility")
            
            # 第三步，确保为常规牙齿生成空间问题（恰好2个不同的牙齿，使用point_2d而非box_2d）
            regular_teeth_qa = []
            if "Teeth visibility with center points" in parsed_data:
                # 强制尝试生成两个问题
                regular_teeth_qa = generate_regular_teeth_spatial_questions(parsed_data, num_questions=2)
                if regular_teeth_qa:
                    qa_pairs.extend(regular_teeth_qa)
                    print(f"  - Added {len(regular_teeth_qa)} spatial questions for regular teeth")
                else:
                    print(f"  - Warning: Could not generate regular teeth spatial questions for {filename}")
            else:
                print(f"  - Warning: No 'Teeth visibility with center points' data in {filename}")
            
            # 记录其他可用类别
            for category in TEMPLATE_DICT:
                if category != "Historical treatments" and category != "Teeth visibility with center points" and category in parsed_data and parsed_data[category]:
                    if category in SPECIAL_CONDITIONS:
                        special_available.append(category)
                    else:
                        regular_available.append(category)
            
            available_categories = special_available + regular_available
            
            print(f"File {filename} has {len(available_categories)} categories: {available_categories}")
            
            # 然后处理其他特殊条件
            for category in special_available:
                # 跳过Historical treatments，因为已经单独处理了
                if category == "Historical treatments":
                    continue
                    
                category_qa = generate_questions_for_category(
                    parsed_data, 
                    category,
                    existing_questions=used_templates,
                    question_types_used=question_types_used,
                    used_items=used_items
                )
                
                if category_qa:
                    qa_pairs.extend(category_qa)
                    print(f"  - Added {len(category_qa)} questions for {category}")
            
            # 最后处理常规类别（除了"Teeth visibility with center points"，已单独处理）
            for category in regular_available:
                if category == "Teeth visibility with center points":
                    continue  # 已经单独处理
                    
                category_qa = generate_questions_for_category(
                    parsed_data, 
                    category,
                    existing_questions=used_templates,
                    question_types_used=question_types_used,
                    used_items=used_items
                )
                
                if category_qa:
                    qa_pairs.extend(category_qa)
                    print(f"  - Added {len(category_qa)} questions for {category}")
            
            # Remove category field from final output
            for qa in qa_pairs:
                if "category" in qa:
                    del qa["category"]
            
            # Create output JSON with question data
            new_data = {
                "image_id": data.get("image_id", ""),
                "file_name": data.get("file_name", ""),
                "image_width": data.get("image_width", 0),
                "image_height": data.get("image_height", 0),
                "sft_data": {
                    "loc_closed_ended": qa_pairs
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            
            # 打印生成结果统计
            teeth_spatial_count = len(regular_teeth_qa)
            has_teeth_count = 1 if teeth_count_qa else 0
            
            print(f"Processed {filename}:")
            print(f"  - Added {len(qa_pairs)} QA pairs total")
            print(f"  - Added {teeth_spatial_count}/2 teeth spatial questions")
            print(f"  - Added {has_teeth_count}/1 teeth count questions")
            print(f"  - Used {len(available_categories) + (1 if has_historical_treatments else 0)} categories")
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    random.seed(42)
    input_folder = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc/"
    output_folder = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc_close_sft_loc/"
    process_json_files(input_folder, output_folder)
