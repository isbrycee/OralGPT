import json
import os
from typing import Dict, Any, List

prefixed_template = "This localization caption provides multi-dimensional spatial analysis of anatomical structures and pathological findings for this panoramic dental X-ray image, including:\n "

def process_bboxes(bbox_data: Any) -> List[List[float]]:
    """处理嵌套的bbox数据结构"""
    bboxes = []
    if isinstance(bbox_data, list):
        if len(bbox_data) == 0:
            return []
        if isinstance(bbox_data[0], list):
            for sublist in bbox_data:
                if len(sublist) >= 4:
                    bboxes.append(sublist[:4])
        else:
            if len(bbox_data) >= 4:
                bboxes.append(bbox_data[:4])
    return bboxes

def generate_section(title: str, items: list, total: bool = True) -> str:
    """生成标准化的章节内容"""
    if not items:
        return ""
    
    section = []
    if total:
        section_title = f"{title} (total: {len(items)}):"
    else:
        section_title = f"{title}:"
    
    # 处理每个item的序列化
    formatted_items = []

    for item in items:
        # 单独序列化每个item
        item_str = json.dumps(item, ensure_ascii=False)
        # 执行必要的字符串替换
        item_str = (
            item_str.replace('"', "'")
            .replace("'present'", "present")
            .replace("'true'", "true")
            .replace("'false'", "false")
        )
        formatted_items.append(f" {item_str}")  # 添加首行缩进
    
    # 构建列表字符串
    items_str = ",\n".join(formatted_items)
    section_content = f"[\n{items_str}\n]"
    
    section.append(section_title)
    section.append(section_content)
    return "\n".join(section)

def generate_loc_caption(data: Dict[str, Any]) -> str:
    """生成完整的location caption"""
    sections = []
    
    # 成像时间
    if data.get("aquisition_time", "N/A") != "N/A":
        sections.append(f"Panoramic Dental X-ray Imaging Time: {data['aquisition_time']}")
    
    # 牙齿可见性
    visibility = []
    for tooth in data["properties"]["Teeth"]:
        if bbox := tooth.get("bbox"):
            if len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                visibility.append({
                    "point_2d": [round(x_center), round(y_center)],
                    "tooth_id": tooth.get("tooth_id", "unknown"),
                    "score": round(tooth.get("score", 0), 2)
                })
    sections.append(generate_section("Teeth visibility with center points", visibility))
    
    # 智齿检测
    wisdom_teeth = []
    for tooth in data["properties"]["Teeth"]:
        if tooth.get("is_wisdom_tooth", False) and (tooth.get("tooth_id") != 'unknown'):
            wisdom_teeth.append({
                "box_2d": [round(coord) for coord in tooth["bbox"]],
                "tooth_id": tooth.get("tooth_id", "unknown"),
                "is_impacted": tooth.get("conditions", {}).get("Impacted tooth", {}).get("present", False),
                "score": round(tooth.get("score", 0), 2)
            })
    sections.append(generate_section("Wisdom teeth detection", wisdom_teeth))
    
    # TODO: Missing teeth
    if "Missing teeth" in data["properties"].keys():
        missing_teeth = []
        for tooth in data["properties"]["Missing teeth"]:
            # 检查是否为缺失牙齿，并确保 tooth_id 有效
            missing_teeth.append({
                "box_2d": tooth.get("bbox", []),  # 兼容可能缺失的 bbox
                "side": tooth.get("side", "unknown"),
                "score": round(tooth.get("score", 0), 2)  # 可选：如果缺失牙齿有置信度评分
            })
        sections.append(generate_section("Missing teeth detection", missing_teeth))

    # 非智齿阻生齿检测（与智齿检测分开处理）
    non_wisdom_impacted = []
    for tooth in data["properties"]["Teeth"]:
        conditions = tooth.get("conditions", {})
        # 检查阻生齿且排除智齿
        if (imp := conditions.get("Impacted tooth", {})) \
            and imp.get("present", False) \
            and not tooth.get("is_wisdom_tooth", False):
            # and (tooth.get("tooth_id") != 'unknown'):
            for bbox in process_bboxes(tooth.get("bbox", [])):
                non_wisdom_impacted.append({
                    "box_2d": [round(coord) for coord in bbox],
                    "tooth_id": tooth.get("tooth_id"),
                    "label": "Impacted tooth",
                    "score": round(imp.get("score", 0), 2)
                })
    sections.append(generate_section("Non-wisdom impacted teeth detection", non_wisdom_impacted))
    
    # 龋齿检测
    caries = []
    for tooth in data["properties"]["Teeth"]:
        conditions = tooth.get("conditions", {})
        for cond in ["Caries", "Deep caries"]:
            if c := conditions.get(cond, {}):
                if c.get("present", False) and (tooth.get("tooth_id") != 'unknown'):
                    caries.append({
                        "box_2d": [round(coord) for coord in c["bbox"]],
                        "tooth_id": tooth.get("tooth_id", "unknown"),
                        "label": cond.replace("_", " "),
                        "score": round(c.get("score", 0), 2)
                    })
                if tooth.get("tooth_id") == 'unknown':
                    for ids, bbox in enumerate(process_bboxes(c.get("bbox", []))):
                        caries.append({
                            "box_2d": [round(coord) for coord in bbox],
                            "tooth_id": tooth.get("tooth_id", "unknown"),
                            "side": c.get("side", 0)[ids] if isinstance(c.get("side", 0), list) else c.get("side", 0),
                            "label": cond.replace("_", " "),
                            "score": c.get("score", 0)[ids] if isinstance(c.get("score", 0), list) else c.get("score", 0),
                        })
    sections.append(generate_section("Dental caries detection", caries))
    
    # 根尖病变检测
    Periapical_lesions = []
    for tooth in data["properties"]["Teeth"]:
        conditions = tooth.get("conditions", {})
        for cond in ["Periapical lesions"]:
            if p := conditions.get(cond, {}):
                if p.get("present", False) and (tooth.get("tooth_id") != 'unknown'):
                    if p.get("score") > 0.7: # filter score lower than 0.7; add by bryce
                        Periapical_lesions.append({
                            "box_2d": [round(coord) for coord in p["bbox"]],
                            "tooth_id": tooth.get("tooth_id", "unknown"),
                            "label": cond + f' ({p.get("type")})' if p.get("type", None) else cond,
                            "score": round(p.get("score", 0), 2)
                        })
                if tooth.get("tooth_id") == 'unknown':
                    for ids, bbox in enumerate(process_bboxes(p.get("bbox", []))):
                        Periapical_lesions.append({
                            "box_2d": [round(coord) for coord in bbox],
                            "tooth_id": tooth.get("tooth_id", "unknown"),
                            "side": p.get("side", 0)[ids] if isinstance(p.get("side", 0), list) else p.get("side", 0),
                            "label": cond.replace("_", " "),
                            "score": p.get("score", 0)[ids] if isinstance(p.get("score", 0), list) else p.get("score", 0),
                        })
                
    sections.append(generate_section("Periapical lesions detection", Periapical_lesions))
    
    # 历史治疗
    treatments = []
    treatment_types = ["Filling", "Crown", "Root canal treatment", "Implant"]
    for tooth in data["properties"]["Teeth"]:
        conditions = tooth.get("conditions", {})
        for tt in treatment_types:
            if t := conditions.get(tt, {}):
                if t.get("present", False) and (tooth.get("tooth_id") != 'unknown'):
                    for ids, bbox in enumerate(process_bboxes(t.get("bbox", []))):
                        treatments.append({
                            "box_2d": [round(coord) for coord in bbox],
                            "tooth_id": tooth.get("tooth_id", "unknown"),
                            "label": tt.replace("_", " "),
                            "score": round(t.get("score", 0)[ids], 2) if isinstance(t.get("score", 0), list) else round(t.get("score", 0), 2)
                        })
                if tooth.get("tooth_id") == 'unknown':
                    for ids, bbox in enumerate(process_bboxes(t.get("bbox", []))):
                        treatments.append({
                            "box_2d": [round(coord) for coord in bbox],
                            "tooth_id": tooth.get("tooth_id", "unknown"),
                            "side": t.get("side", 0)[ids] if isinstance(t.get("side", 0), list) else t.get("side", 0),
                            "label": tt.replace("_", " "),
                            "score": t.get("score", 0)[ids] if isinstance(t.get("score", 0), list) else t.get("score", 0),
                        })

    sections.append(generate_section("Historical treatments", treatments))
    
    # 颌骨相关检测
    jawbone_sections = []
    for jawbone in data["properties"].get("JawBones", []):
        conditions = jawbone.get("conditions", {})
        
        # 骨质流失
        if bone_loss := conditions.get("Bone loss", {}):
            for ids, bbox in enumerate(process_bboxes(bone_loss.get("bbox", []))):
                jawbone_sections.append({
                    "box_2d": [round(coord) for coord in bbox],
                    "label": "Bone loss",
                    "side": bone_loss.get("side", 0)[ids] if isinstance(bone_loss.get("side", 0), list) else bone_loss.get("side", 0),
                    "score": round(bone_loss.get("score", 0)[ids], 2) if isinstance(bone_loss.get("score", 0), list) else round(bone_loss.get("score", 0), 2)
                })
        
        # 下颌管
        if canal := conditions.get("Mandibular canal", {}):
            for ids, bbox in enumerate(process_bboxes(canal.get("bbox", []))):
                jawbone_sections.append({
                    "box_2d": [round(coord) for coord in bbox],
                    "label": "Mandibular canal",
                    "score": round(canal.get("score", 0)[ids], 2) if isinstance(canal.get("score", 0), list) else round(canal.get("score", 0), 2)
                })
        
        # 上颌窦
        if sinuses := conditions.get("Maxillary sinus", {}):
            for ids, bbox in enumerate(process_bboxes(sinuses.get("bbox", []))):
                jawbone_sections.append({
                    "box_2d": [round(coord) for coord in bbox],
                    "label": "Maxillary sinus",
                    "score": round(sinuses.get("score", 0)[ids], 2) if isinstance(sinuses.get("score", 0), list) else round(sinuses.get("score", 0), 2)
                })
    
    sections.append(generate_section("Bone loss detection", 
                   [x for x in jawbone_sections if x["label"] == "Bone loss"]))
    sections.append(generate_section("Mandibular canal visibility", 
                   [x for x in jawbone_sections if x["label"] == "Mandibular canal"]))
    sections.append(generate_section("Maxillary sinuses visibility", 
                   [x for x in jawbone_sections if x["label"] == "Maxillary sinus"]))
    
    return "\n\n".join([s for s in sections if s])

def process_json_file(file_path: str, saved_folder: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["loc_caption"] = prefixed_template + generate_loc_caption(data)
    # print(file_path)
    # print(data["med_report"])
    file_name = file_path.split('/')[-1]
    with open(os.path.join(saved_folder, file_name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(input_folder: str, saved_folder: str):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".json"):
                process_json_file(os.path.join(root, file), saved_folder)

if __name__ == "__main__":
    input_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report/"
    saved_folder = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report_sft"

    os.makedirs(saved_folder, exist_ok=True)

    main(input_folder, saved_folder)