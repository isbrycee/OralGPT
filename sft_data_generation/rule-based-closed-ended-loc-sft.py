import json
import os
import re
import random
from ast import literal_eval
import copy
from typing import List, Dict, Any, Tuple

# Comprehensive template dictionaries for each category
TEMPLATE_DICT = {
    "Teeth visibility with center points": [
        "How many teeth are visible in this panoramic dental X-ray image?",
        "What is the total number of visible teeth in this panoramic image?",
        "Count the number of teeth that can be seen in this dental X-ray.",
        "Which tooth is located at coordinates [{}, {}] in the panoramic image?",
        "Identify the position of tooth #{} in this X-ray image.",
        "Output the coordinates of tooth #{} in the panoramic dental X-ray."
    ],
    "Wisdom teeth detection": [
        "What is the count of wisdom teeth visible in this X-ray?",
        "How many wisdom teeth can be identified in this panoramic image?",
        "Where is the wisdom tooth #{} located in this panoramic X-ray?",
        "Which wisdom teeth are impacted in this panoramic dental X-ray?",
        "Are any wisdom teeth fully erupted in this X-ray image?"
    ],
    "Missing teeth detection": [
        "How many teeth are missing in this panoramic dental X-ray?",
        "Count the number of missing teeth in this dental X-ray.",
        "In which part of the dental arch are teeth missing in this X-ray?",
        "Which teeth are missing in this panoramic dental image?",
        "Where are gaps from missing teeth located in this panoramic image?"
    ],
    "Non-wisdom impacted teeth detection": [
        "How many non-wisdom impacted teeth are visible in this panoramic X-ray?",
        "Where is the impacted non-wisdom tooth located in this panoramic X-ray?",
        "Which teeth are impacted (excluding wisdom teeth) in this dental image?",
        "Are there any impacted canines visible in this panoramic X-ray?"
    ],
    "Dental caries detection": [
        "How many teeth with caries are detected in this panoramic image?",
        "Which teeth are affected by caries in this panoramic dental image?",
        "Where are the caries detected in this X-ray image?",
        "Which teeth have deep caries in this panoramic dental X-ray?",
        "What is the distribution of caries in this dental X-ray?"
    ],
    "Periapical lesions detection": [
        "How many teeth show periapical lesions in this panoramic X-ray?",
        "Where is a periapical lesion located in this panoramic X-ray?",
        "Which teeth are affected by periapical lesions in this panoramic X-ray?",
        "Are there any periapical lesions visible in this dental image?",
        "What is the size of the periapical lesion detected in this X-ray?"
    ],
    "Historical treatments": [
        "How many dental fillings can be detected in this panoramic image?",
        "Which teeth have dental restorations in this panoramic dental image?",
        "What types of dental treatments are visible in this panoramic X-ray?",
        "Where are dental restorations detected in this panoramic image?",
        "Which teeth show evidence of root canal treatment in this panoramic image?"
    ],
    "Bone loss detection": [
        "How many areas of bone loss can be identified in this panoramic image?",
        "Where is bone loss detected in this panoramic X-ray?",
        "On which side(s) is bone loss detected in this panoramic X-ray?",
        "What is the severity of bone loss visible in this dental X-ray?",
        "Which teeth are affected by adjacent bone loss in this image?"
    ],
    "Mandibular canal visibility": [
        "Is the mandibular canal visible in this panoramic X-ray?",
        "On which sides is the mandibular canal visible in this panoramic X-ray?",
        "Where is the mandibular canal located in this panoramic X-ray?",
        "How clear is the visualization of the mandibular canal in this image?",
        "What is the relationship between the mandibular canal and adjacent teeth roots?"
    ],
    "Maxillary sinuses visibility": [
        "Are the maxillary sinuses visible in this panoramic X-ray?",
        "Where are the maxillary sinuses located in this panoramic X-ray?",
        "Is there any pathology visible in the maxillary sinuses in this X-ray?",
        "How well can you visualize the maxillary sinuses in this panoramic image?",
        "What is the relationship between the maxillary sinuses and the upper molar roots?"
    ]
}

# Keep the existing specialized templates for backwards compatibility
COUNTING_TEMPLATES = TEMPLATE_DICT["Teeth visibility with center points"][:3] + [
    TEMPLATE_DICT["Dental caries detection"][0],
    TEMPLATE_DICT["Wisdom teeth detection"][0],
    TEMPLATE_DICT["Historical treatments"][4],
    TEMPLATE_DICT["Missing teeth detection"][1],
    TEMPLATE_DICT["Bone loss detection"][0],
    "What is the total number of impacted teeth shown in this X-ray?",
    TEMPLATE_DICT["Historical treatments"][0]
]

TOOTH_LOCALIZATION_TEMPLATES = TEMPLATE_DICT["Teeth visibility with center points"][3:6] + [
    "Where is tooth #{} located in this panoramic dental image?",
    "Identify the position of tooth #{} in this X-ray image.",
    "Output the coordinates of tooth #{} in the panoramic dental X-ray.",
    "Detect the location of the {} in this panoramic image.",
    "Which tooth is positioned at the center point [{}, {}] in this dental X-ray?",
    "Locate tooth #{} in this panoramic radiograph.",
    "Where are the wisdom teeth positioned in this dental panoramic image?",
    "Identify the location of tooth #{} in this X-ray image."
]

PATHOLOGY_TEMPLATES = [
    "Which teeth are affected by {} in this panoramic dental image?",
    "Where are the {} detected in this X-ray image?",
    "Output the positions of {} in the panoramic X-ray image.",
    "Which teeth show evidence of {} in this panoramic image?",
    "Identify the areas showing {} in the panoramic radiograph.",
    "Detect the locations of {} in this X-ray image.",
    "Which teeth have {} in this panoramic dental image?",
    "Where are the {} located in this X-ray?",
    "Identify which teeth exhibit {} in this panoramic image.",
    "Output the locations of all pathological findings in this dental X-ray."
]

TREATMENT_TEMPLATES = [
    "Which teeth have {} in this panoramic dental image?",
    "Identify the teeth with {} in this X-ray.",
    "Where are {} detected in this panoramic image?",
    "Which teeth show evidence of {} in this X-ray?",
    "Output the positions of all dental restorations in the panoramic image.",
    "Detect the teeth with historical treatments in this dental X-ray.",
    "Which dental treatments are visible in the area [x1, y1, x2, y2]?",
    "Identify all teeth with {} in this panoramic radiograph.",
    "What type of dental treatment is shown in the bounding box [x1, y1, x2, y2]?",
    "Which teeth have undergone {} in this panoramic image?"
]

ANATOMY_TEMPLATES = [
    "Which anatomical structures are visible in the panoramic dental image?",
    "Where is the {} located in this X-ray image?",
    "Identify the {} in this panoramic radiograph.",
    "Detect the locations of bone structures in this dental X-ray.",
    "Output the positions of the {} in the panoramic image.",
    "Which anatomical landmarks have been accurately detected in the panoramic image?",
    "Please detect the {} in this X-ray image.",
    "Which areas show the {} in the panoramic image?",
    "Identify areas showing the {} in the panoramic radiograph.",
    "Output the boundaries of the {} in this dental X-ray."
]

MISSING_TEETH_TEMPLATES = [
    "Which teeth are missing in this panoramic dental image?",
    "Identify the missing teeth in this X-ray.",
    "Where are gaps from missing teeth located in this panoramic image?",
    "Which areas show evidence of missing teeth in this dental X-ray?",
    "Output the positions of dental gaps in the panoramic image.",
    "Detect the missing teeth in the upper arch of this X-ray.",
    "Which teeth are absent in the lower jaw of this panoramic image?",
    "Identify all missing teeth positions in this dental radiograph.",
    "Which tooth numbers are missing in this panoramic X-ray?",
    "In which regions of the dental arches are teeth missing in this image?"
]

def parse_medical_string(input_str):
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

def generate_qa_for_category(category, templates, parsed_data, max_questions=3):
    """Generate questions and answers for a specific category using templates"""
    if category not in parsed_data or not parsed_data[category]:
        return []
    
    qa_pairs = []
    category_data = parsed_data[category]
    
    # Shuffle templates to get variety
    shuffled_templates = random.sample(templates, min(len(templates), max_questions))
    
    for template in shuffled_templates:
        # Skip if we've reached max questions
        if len(qa_pairs) >= max_questions:
            break
            
        # For counting questions
        if any(keyword in template.lower() for keyword in ["how many", "count", "number", "total"]):
            count = len(category_data)
            wrong_counts = generate_wrong_count(count)
            options, correct_letter = generate_multiple_choice_options(count, wrong_counts)
            
            qa_pairs.append({
                "question": template,
                "options": options,
                "answer": correct_letter,
                "explanation": f"There are {count} {category.lower().replace('detection', '').strip()} in the panoramic X-ray."
            })
            continue
            
        # For location/where questions
        if any(keyword in template.lower() for keyword in ["where", "locat", "position", "coordinat"]):
            if not category_data:
                continue
                
            sample_item = random.choice(category_data)
            
            # Handle template formatting with tooth ID or coordinates
            if "{}" in template:
                if "tooth_id" in sample_item and template.count("{}") == 1:
                    question = template.format(sample_item["tooth_id"])
                elif "point_2d" in sample_item and template.count("{}") == 2:
                    x, y = sample_item["point_2d"]
                    question = template.format(x, y)
                else:
                    continue
            else:
                question = template
            
            # Generate answer based on available data
            if "box_2d" in sample_item:
                correct_bbox = sample_item["box_2d"]
                correct_bbox_str = str(correct_bbox)
                
                # Generate wrong bounding boxes
                wrong_bboxes = []
                for _ in range(3):
                    wrong_bboxes.append(str(generate_wrong_bbox(correct_bbox)))
                
                options, correct_letter = generate_multiple_choice_options(correct_bbox_str, wrong_bboxes)
                
                explanation = f"The {category.lower().replace('detection', '').strip()} is located at the bounding box {correct_bbox_str}."
                if "tooth_id" in sample_item:
                    explanation = f"Tooth #{sample_item['tooth_id']} is located at the bounding box {correct_bbox_str}."
                
                qa_pairs.append({
                    "question": question,
                    "options": options,
                    "answer": correct_letter,
                    "explanation": explanation
                })
                continue
                
        # For which teeth/affected questions
        if any(keyword in template.lower() for keyword in ["which teeth", "affected", "exhibit"]):
            affected_teeth = [item.get("tooth_id") for item in category_data 
                             if "tooth_id" in item and item["tooth_id"] != "unknown"]
            
            if affected_teeth:
                correct_answer = ", ".join([f"#{t}" for t in affected_teeth])
                
                # Generate wrong options
                if "Teeth visibility with center points" in parsed_data:
                    all_teeth = [t.get("tooth_id") for t in parsed_data["Teeth visibility with center points"]
                                if "tooth_id" in t and t["tooth_id"] != "unknown"]
                    non_affected_teeth = [t for t in all_teeth if t not in affected_teeth]
                    
                    wrong_options = []
                    for _ in range(3):
                        if non_affected_teeth:
                            sample_size = min(len(affected_teeth), len(non_affected_teeth))
                            if sample_size > 0:
                                wrong_sample = random.sample(non_affected_teeth, sample_size)
                                wrong_options.append(", ".join([f"#{t}" for t in wrong_sample]))
                            else:
                                wrong_options.append("No teeth affected")
                        else:
                            wrong_options.append("No teeth affected")
                else:
                    wrong_options = ["No teeth affected", "All teeth affected", "Cannot be determined"]
                
                options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                
                explanation = f"The teeth with {category.lower().replace('detection', '').strip()} are: {correct_answer}."
                
                qa_pairs.append({
                    "question": template,
                    "options": options,
                    "answer": correct_letter,
                    "explanation": explanation
                })
                continue
                
        # For yes/no questions
        if any(keyword in template.lower() for keyword in ["is ", "are ", "visible"]):
            if category_data:
                correct_answer = "Yes"
                explanation = f"The {category.lower().replace('detection', '').replace('visibility', '').strip()} is visible in the panoramic X-ray."
                wrong_options = ["No", "Partially visible", "Cannot be determined"]
            else:
                correct_answer = "No"
                explanation = f"The {category.lower().replace('detection', '').replace('visibility', '').strip()} is not visible in the panoramic X-ray."
                wrong_options = ["Yes", "Partially visible", "Cannot be determined"]
            
            options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
            
            qa_pairs.append({
                "question": template,
                "options": options,
                "answer": correct_letter,
                "explanation": explanation
            })
            continue
            
        # For side/region questions
        if any(keyword in template.lower() for keyword in ["side", "region", "part"]):
            sides = {item.get("side", "unknown") for item in category_data if "side" in item}
            
            if sides and "unknown" not in sides:
                correct_answer = ", ".join(s.capitalize() for s in sides)
                
                # Generate wrong options
                all_sides = ["upper", "lower", "left", "right"]
                remaining_sides = [s for s in all_sides if s not in sides]
                
                wrong_options = []
                if remaining_sides:
                    wrong_options.append(random.choice(remaining_sides).capitalize())
                if len(sides) < len(all_sides):
                    wrong_options.append("All sides")
                wrong_options.append(f"No {category.lower().replace('detection', '').strip()} detected")
                
                # Ensure we have 3 wrong options
                while len(wrong_options) < 3:
                    wrong_options.append("Cannot be determined")
                
                options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options)
                
                explanation = f"The {category.lower().replace('detection', '').strip()} is detected on the {correct_answer} side(s)."
                
                qa_pairs.append({
                    "question": template,
                    "options": options,
                    "answer": correct_letter,
                    "explanation": explanation
                })
                continue
    
    # Return unique questions
    unique_qa_pairs = []
    seen_questions = set()
    
    for qa in qa_pairs:
        if qa["question"] not in seen_questions:
            seen_questions.add(qa["question"])
            unique_qa_pairs.append(qa)
    
    return unique_qa_pairs

# Keep the original functions for backwards compatibility
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
        
        # Filter templates based on their placeholder requirements
        templates_for_coordinates = []
        templates_for_tooth_id = []
        
        for template in TOOTH_LOCALIZATION_TEMPLATES:
            placeholders = template.count("{}")
            if placeholders == 2 and "center point" in template:
                templates_for_coordinates.append(template)
            elif placeholders == 1 and "#" not in template:  # Ensure no # character in template
                templates_for_tooth_id.append(template)
        
        # Generate a coordinate-based question (which tooth is at position X,Y)
        if templates_for_coordinates:
            template = random.choice(templates_for_coordinates)
            question = template.format(x, y)
            correct_answer = str(tooth_id)
            
            # Generate wrong options (other tooth IDs)
            wrong_options = [str(t["tooth_id"]) for t in teeth_data if t["tooth_id"] != tooth_id]
            if len(wrong_options) < 3:
                # Add some random tooth numbers if we don't have enough
                potential_ids = [str(i) for i in range(11, 49) if str(i) not in wrong_options and str(i) != tooth_id]
                wrong_options.extend(random.sample(potential_ids, min(3 - len(wrong_options), len(potential_ids))))
            
            options, correct_letter = generate_multiple_choice_options(correct_answer, wrong_options[:3])
            
            qa_pairs.append({
                "question": question,
                "options": options,
                "answer": correct_letter,
                "explanation": f"The tooth at center point [{x}, {y}] is tooth #{tooth_id}."
            })
        
        # Generate a tooth-id based question (where is tooth #X located)
        if templates_for_tooth_id:
            template = random.choice(templates_for_tooth_id)
            question = template.format(f"#{tooth_id}")  # Manually adding # symbol
            
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

def process_json_files(input_folder, output_folder):
    """Process all JSON files in the input folder and add QA data"""
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # 确保类别名称与生成数据时完全一致
    all_possible_categories = [
        "Teeth visibility with center points",
        "Wisdom teeth detection",
        "Missing teeth detection",
        "Non-wisdom impacted teeth detection",  # 修正为完整名称
        "Dental caries detection",
        "Periapical lesions detection",
        "Historical treatments",
        "Bone loss detection",
        "Mandibular canal visibility",
        "Maxillary sinuses visibility"
    ]
    
    # 每个类别的关键词，用于确保问题文本中包含相关关键词
    category_keywords = {
        "Teeth visibility with center points": ["teeth", "tooth", "visible"],
        "Wisdom teeth detection": ["wisdom"],
        "Missing teeth detection": ["missing"],
        "Non-wisdom impacted teeth detection": ["impacted", "non-wisdom"],
        "Dental caries detection": ["caries", "decay", "cavities"],
        "Periapical lesions detection": ["periapical", "lesion"],
        "Historical treatments": ["treatment", "filling", "restoration", "root canal"],
        "Bone loss detection": ["bone loss"],
        "Mandibular canal visibility": ["mandibular", "canal"],
        "Maxillary sinuses visibility": ["maxillary", "sinus", "sinuses"]
    }
    
    # 处理每个JSON文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # 读取输入JSON
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查loc_caption是否存在
                if "loc_caption" not in data:
                    print(f"Skipping {filename}: No localization caption found")
                    continue
                
                try:
                    # 提取并解析定位说明
                    if "including:\n" in data["loc_caption"]:
                        loc_caption = data["loc_caption"].split('including:\n')[1].strip()
                        parsed_data = parse_medical_string(loc_caption)
                        
                        # 初始化问答对列表
                        qa_pairs = []
                        
                        # 确定此图像中可用的类别
                        # 重要：确保类别名称完全匹配
                        available_categories = []
                        for category in all_possible_categories:
                            if category in parsed_data and parsed_data[category]:
                                available_categories.append(category)
                        
                        print(f"File {filename} has {len(available_categories)} categories: {available_categories}")
                        
                        # 跟踪每个类别已生成的问题
                        category_questions = {category: [] for category in available_categories}
                        
                        # 阶段1：为每个可用类别生成至少1个问题
                        for category in available_categories:
                            if category in TEMPLATE_DICT:
                                # 使用特定类别的模板
                                templates = TEMPLATE_DICT[category]
                                
                                # 尝试3次为此类别生成有效问题
                                for attempt in range(3):
                                    # 为此类别生成1个问答对
                                    category_qa = generate_qa_for_category(category, templates, parsed_data, max_questions=1)
                                    
                                    if category_qa:
                                        # 为每个问题添加类别标签以便跟踪
                                        for qa in category_qa:
                                            qa["category"] = category
                                            
                                            # 确保问题文本中包含类别关键词
                                            keywords = category_keywords.get(category, [category.split()[0].lower()])
                                            question_text = qa["question"].lower()
                                            
                                            # 如果问题中不包含任何关键词，则修改问题
                                            if not any(keyword.lower() in question_text for keyword in keywords):
                                                # 添加类别前缀以确保可识别
                                                qa["question"] = f"Regarding {category.lower()}: {qa['question']}"
                                                print(f"  - Modified question to include category: {qa['question']}")
                                        
                                        qa_pairs.extend(category_qa)
                                        category_questions[category].extend(category_qa)
                                        print(f"  - Added initial question for {category}")
                                        break
                        
                        # 阶段2：填充问题数量至10个，优先考虑问题较少的类别
                        while len(qa_pairs) < 10:
                            # 按问题数量排序类别（优先使用问题最少的类别）
                            sorted_categories = sorted(
                                available_categories, 
                                key=lambda cat: len(category_questions.get(cat, []))
                            )
                            
                            # 如果所有类别至少有2个问题，则停止
                            if all(len(questions) >= 2 for questions in category_questions.values()):
                                break
                                
                            # 获取问题最少的类别
                            if not sorted_categories:
                                break
                                
                            current_category = sorted_categories[0]
                            
                            if current_category in TEMPLATE_DICT:
                                templates = TEMPLATE_DICT[current_category]
                                
                                # 为此类别再生成1个问题
                                additional_qa = generate_qa_for_category(
                                    current_category, templates, parsed_data, max_questions=1
                                )
                                
                                if additional_qa:
                                    # 确保不添加重复问题
                                    existing_questions = {qa["question"] for qa in qa_pairs}
                                    new_qa = [qa for qa in additional_qa 
                                             if qa["question"] not in existing_questions]
                                    
                                    # 添加类别标签并确保问题中包含关键词
                                    for qa in new_qa:
                                        qa["category"] = current_category
                                        
                                        # 确保问题文本中包含类别关键词
                                        keywords = category_keywords.get(current_category, [current_category.split()[0].lower()])
                                        question_text = qa["question"].lower()
                                        
                                        # 如果问题中不包含任何关键词，则修改问题
                                        if not any(keyword.lower() in question_text for keyword in keywords):
                                            # 添加类别前缀以确保可识别
                                            qa["question"] = f"Regarding {current_category.lower()}: {qa['question']}"
                                            print(f"  - Modified question to include category: {qa['question']}")
                                    
                                    if new_qa:
                                        qa_pairs.extend(new_qa)
                                        category_questions[current_category].extend(new_qa)
                                        print(f"  - Added additional question for {current_category}")
                                    else:
                                        # 如果无法为此类别添加更多唯一问题，则下次跳过它
                                        print(f"  - No more unique questions for {current_category}")
                                        # 添加一个占位符问题，以避免再次尝试此类别
                                        category_questions[current_category].append(None)
                                else:
                                    # 如果无法为此类别生成更多问题，则下次跳过它
                                    print(f"  - Failed to generate more questions for {current_category}")
                                    # 添加一个占位符问题，以避免再次尝试此类别
                                    category_questions[current_category].append(None)
                            else:
                                # 没有合适的模板，跳过此类别
                                category_questions[current_category].append(None)
                        
                        # 阶段3：如果仍然没有10个问题，尝试使用专门的函数
                        if len(qa_pairs) < 10:
                            # 移除category_questions中的None条目
                            for category in category_questions:
                                category_questions[category] = [q for q in category_questions[category] if q is not None]
                            
                            # 尝试使用专门的函数生成更多问题
                            additional_qa = []
                            
                            # 专注于问题最少的前3个类别
                            categories_to_boost = sorted(
                                available_categories, 
                                key=lambda cat: len(category_questions.get(cat, []))
                            )[:3]
                            
                            for category in categories_to_boost:
                                # 为每种类别使用专门的函数
                                try:
                                    if category == "Teeth visibility with center points":
                                        counting_qa = generate_counting_qa(parsed_data)
                                        # 标记这些问题
                                        for qa in counting_qa:
                                            qa["category"] = category
                                            # 确保问题文本中包含类别关键词
                                            keywords = category_keywords.get(category, [])
                                            if not any(keyword.lower() in qa["question"].lower() for keyword in keywords):
                                                qa["question"] = f"Regarding {category.lower()}: {qa['question']}"
                                        additional_qa.extend(counting_qa)
                                        
                                        try:
                                            loc_qa = generate_tooth_localization_qa(parsed_data)
                                            # 标记这些问题
                                            for qa in loc_qa:
                                                qa["category"] = category
                                                # 确保问题文本中包含类别关键词
                                                keywords = category_keywords.get(category, [])
                                                if not any(keyword.lower() in qa["question"].lower() for keyword in keywords):
                                                    qa["question"] = f"Regarding {category.lower()}: {qa['question']}"
                                            additional_qa.extend(loc_qa)
                                        except Exception as e:
                                            print(f"  - Error in tooth localization: {str(e)}")
                                            
                                    elif category == "Dental caries detection":
                                        pathology_qa = generate_pathology_qa(parsed_data)
                                        # 标记并过滤此类别的问题
                                        for qa in pathology_qa:
                                            if "caries" in qa["question"].lower():
                                                qa["category"] = category
                                                additional_qa.append(qa)
                                    
                                    elif category == "Periapical lesions detection":
                                        pathology_qa = generate_pathology_qa(parsed_data)
                                        # 标记并过滤此类别的问题
                                        for qa in pathology_qa:
                                            if "periapical" in qa["question"].lower():
                                                qa["category"] = category
                                                additional_qa.append(qa)
                                    
                                    elif category == "Historical treatments":
                                        treatment_qa = generate_treatment_qa(parsed_data)
                                        # 标记这些问题
                                        for qa in treatment_qa:
                                            qa["category"] = category
                                            # 确保问题文本中包含类别关键词
                                            keywords = category_keywords.get(category, [])
                                            if not any(keyword.lower() in qa["question"].lower() for keyword in keywords):
                                                qa["question"] = f"Regarding {category.lower()}: {qa['question']}"
                                        additional_qa.extend(treatment_qa)
                                    
                                    elif category == "Missing teeth detection":
                                        missing_qa = generate_missing_teeth_qa(parsed_data)
                                        # 标记这些问题
                                        for qa in missing_qa:
                                            qa["category"] = category
                                            # 确保问题文本中包含类别关键词
                                            keywords = category_keywords.get(category, [])
                                            if not any(keyword.lower() in qa["question"].lower() for keyword in keywords):
                                                qa["question"] = f"Regarding {category.lower()}: {qa['question']}"
                                        additional_qa.extend(missing_qa)
                                    
                                    elif category == "Mandibular canal visibility":
                                        anatomy_qa = generate_anatomy_qa(parsed_data)
                                        # 标记并过滤此类别的问题
                                        for qa in anatomy_qa:
                                            if "mandibular" in qa["question"].lower():
                                                qa["category"] = category
                                                additional_qa.append(qa)
                                    
                                    elif category == "Maxillary sinuses visibility":
                                        anatomy_qa = generate_anatomy_qa(parsed_data)
                                        # 标记并过滤此类别的问题
                                        for qa in anatomy_qa:
                                            if "maxillary" in qa["question"].lower() or "sinus" in qa["question"].lower():
                                                qa["category"] = category
                                                additional_qa.append(qa)
                                                
                                    elif category == "Non-wisdom impacted teeth detection":
                                        # 为非智齿阻生齿生成专门的问题
                                        impacted_qa = []
                                        for template in TEMPLATE_DICT.get(category, []):
                                            qa = generate_qa_for_category(category, [template], parsed_data, max_questions=1)
                                            if qa:
                                                for q in qa:
                                                    q["category"] = category
                                                    # 确保问题文本包含关键词
                                                    if "impacted" not in q["question"].lower():
                                                        q["question"] = f"Regarding non-wisdom impacted teeth: {q['question']}"
                                                impacted_qa.extend(qa)
                                        additional_qa.extend(impacted_qa)
                                    
                                    elif category == "Bone loss detection":
                                        # 为骨质流失生成专门的问题
                                        bone_qa = []
                                        for template in TEMPLATE_DICT.get(category, []):
                                            qa = generate_qa_for_category(category, [template], parsed_data, max_questions=1)
                                            if qa:
                                                for q in qa:
                                                    q["category"] = category
                                                    # 确保问题文本包含关键词
                                                    if "bone loss" not in q["question"].lower():
                                                        q["question"] = f"Regarding bone loss: {q['question']}"
                                                bone_qa.extend(qa)
                                        additional_qa.extend(bone_qa)
                                        
                                    # 如果已添加足够的问题，则停止
                                    if len(qa_pairs) + len(additional_qa) >= 10:
                                        break
                                except Exception as e:
                                    print(f"  - Error generating questions for {category}: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # 确保问题文本中包含类别关键词
                            for qa in additional_qa:
                                if "category" in qa:
                                    category = qa["category"]
                                    keywords = category_keywords.get(category, [category.split()[0].lower()])
                                    question_text = qa["question"].lower()
                                    
                                    if not any(keyword.lower() in question_text for keyword in keywords):
                                        qa["question"] = f"Regarding {category.lower()}: {qa['question']}"
                            
                            # 过滤掉已有的问题
                            existing_questions = {qa["question"] for qa in qa_pairs}
                            new_qa = [qa for qa in additional_qa if qa["question"] not in existing_questions]
                            
                            # 添加直到达到10个
                            if new_qa:
                                # 优先添加来自问题较少类别的问题
                                new_qa.sort(key=lambda qa: len(category_questions.get(qa.get("category", ""), [])))
                                to_add = new_qa[:min(10 - len(qa_pairs), len(new_qa))]
                                
                                for qa in to_add:
                                    qa_pairs.append(qa)
                                    if "category" in qa:
                                        category_questions[qa["category"]].append(qa)
                        
                        # 过滤掉空类别和None值
                        for category in list(category_questions.keys()):
                            category_questions[category] = [q for q in category_questions[category] if q is not None]
                            if not category_questions[category]:
                                del category_questions[category]
                        
                        # 创建新的JSON对象，仅包含所需字段
                        new_data = {
                            "image_id": data.get("image_id", ""),
                            "file_name": data.get("file_name", ""),
                            "image_width": data.get("image_width", 0),
                            "image_height": data.get("image_height", 0)
                        }
                        
                        # 从最终输出中删除类别标签
                        for qa in qa_pairs:
                            if "category" in qa:
                                del qa["category"]
                        
                        # 添加SFT数据
                        new_data["sft_data"] = {}
                        new_data["sft_data"]["loc_closed_ended"] = qa_pairs
                        
                        # 将更新后的JSON写入输出文件夹
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(new_data, f, indent=2, ensure_ascii=False)
                        
                        # 统计每个类别的问题数量
                        category_counts = {cat: len(qs) for cat, qs in category_questions.items()}
                        
                        # 检查是否所有类别都被使用
                        unused_categories = [cat for cat in available_categories if cat not in category_counts]
                        
                        print(f"Processed {filename}:")
                        print(f"  - Added {len(qa_pairs)} QA pairs")
                        print(f"  - Used {len(category_counts)}/{len(available_categories)} feature categories")
                        
                        if unused_categories:
                            print(f"  - WARNING: {len(unused_categories)} categories were not used: {unused_categories}")
                        
                        for cat, count in category_counts.items():
                            print(f"  - {cat}: {count} questions")
                    else:
                        print(f"Skipping {filename}: Invalid localization caption format")
                        
                except Exception as e:
                    print(f"Error parsing localization caption in {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define input and output folders
    input_folder = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc/"
    output_folder = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc_close_sft_loc/"
    
    # Process all JSON files
    process_json_files(input_folder, output_folder)
