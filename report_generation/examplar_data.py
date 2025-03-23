example_input = """
This localization caption provides multi-dimensional spatial analysis of anatomical structures and pathological findings for this panoramic dental X-ray image, including:

Teeth visibility with center points (total: 30):
[
 {'point_2d': [680, 403], 'tooth_id': '21', 'score': 0.9},
 {'point_2d': [662, 499], 'tooth_id': '31', 'score': 0.89},
 {'point_2d': [709, 385], 'tooth_id': '22', 'score': 0.88},
 {'point_2d': [611, 391], 'tooth_id': '12', 'score': 0.87},
 {'point_2d': [530, 349], 'tooth_id': '16', 'score': 0.86},
 {'point_2d': [644, 405], 'tooth_id': '11', 'score': 0.85},
 {'point_2d': [553, 480], 'tooth_id': '46', 'score': 0.84},
 {'point_2d': [775, 490], 'tooth_id': '35', 'score': 0.81},
 {'point_2d': [639, 502], 'tooth_id': '41', 'score': 0.79},
 {'point_2d': [468, 339], 'tooth_id': '17', 'score': 0.75},
 {'point_2d': [491, 451], 'tooth_id': '47', 'score': 0.73},
 {'point_2d': [562, 495], 'tooth_id': '45', 'score': 0.7},
 {'point_2d': [734, 512], 'tooth_id': '34', 'score': 0.69},
 {'point_2d': [883, 345], 'tooth_id': '26', 'score': 0.68},
 {'point_2d': [616, 512], 'tooth_id': '42', 'score': 0.67},
 {'point_2d': [595, 392], 'tooth_id': '13', 'score': 0.67},
 {'point_2d': [598, 518], 'tooth_id': '44', 'score': 0.65},
 {'point_2d': [423, 338], 'tooth_id': '18', 'score': 0.65},
 {'point_2d': [992, 407], 'tooth_id': '38', 'score': 0.59},
 {'point_2d': [938, 320], 'tooth_id': '28', 'score': 0.56},
 {'point_2d': [697, 509], 'tooth_id': '33', 'score': 0.56},
 {'point_2d': [854, 497], 'tooth_id': '35', 'score': 0.55},
 {'point_2d': [549, 366], 'tooth_id': '15', 'score': 0.5},
 {'point_2d': [933, 437], 'tooth_id': '36', 'score': 0.5},
 {'point_2d': [831, 349], 'tooth_id': '25', 'score': 0.39},
 {'point_2d': [854, 497], 'tooth_id': '36', 'score': 0.37},
 {'point_2d': [784, 373], 'tooth_id': '24', 'score': 0.35},
 {'point_2d': [992, 407], 'tooth_id': '37', 'score': 0.32},
 {'point_2d': [933, 437], 'tooth_id': '37', 'score': 0.3},
 {'point_2d': [746, 372], 'tooth_id': '23', 'score': 0.3}
]

Wisdom teeth detection (total: 3):
[
 {'box_2d': [730, 446, 115, 230], 'tooth_id': '18', 'is_impacted': false, 'score': 0.65},
 {'box_2d': [1782, 648, 202, 166], 'tooth_id': '38', 'is_impacted': false, 'score': 0.59},
 {'box_2d': [1748, 430, 128, 210], 'tooth_id': '28', 'is_impacted': true, 'score': 0.56}
]

Dental caries detection (total: 3):
[
 {'box_2d': [772, 496, 17, 43], 'tooth_id': '18', 'label': 'Caries', 'score': 0.34},
 {'box_2d': [1790, 657, 245, 203], 'tooth_id': '38', 'label': 'Deep caries', 'score': 0.46}
]

Periapical lesions detection (total: 1):
[
 {'box_2d': [1497, 877, 48, 48], 'tooth_id': '35', 'label': 'Periapical lesions (Granuloma)', 'score': 0.84}
]

Historical treatments (total: 12):
[
 {'box_2d': [953, 558, 45, 26], 'tooth_id': '16', 'label': 'Filling', 'score': 0.7},
 {'box_2d': [941, 688, 33, 27], 'tooth_id': '46', 'label': 'Filling', 'score': 0.7},
 {'box_2d': [1452, 707, 57, 30], 'tooth_id': '35', 'label': 'Filling', 'score': 0.35},
 {'box_2d': [782, 522, 20, 49], 'tooth_id': '18', 'label': 'Filling', 'score': 0.39},
 {'box_2d': [1751, 597, 39, 37], 'tooth_id': '38', 'label': 'Filling', 'score': 0.81},
 {'box_2d': [1542, 709, 101, 81], 'tooth_id': '35', 'label': 'Crown', 'score': 0.65},
 {'box_2d': [1590, 815, 85, 147], 'tooth_id': '35', 'label': 'Implant', 'score': 0.81},
 {'box_2d': [992, 558, 19, 35], 'tooth_id': '15', 'label': 'Filling', 'score': 0.43},
 {'box_2d': [1557, 553, 46, 52], 'tooth_id': '25', 'label': 'Filling', 'score': 0.6},
 {'box_2d': [1586, 479, 29, 99], 'tooth_id': '25', 'label': 'Root canal treatment', 'score': 0.73},
 {'box_2d': [1502, 557, 31, 65], 'tooth_id': '24', 'label': 'Filling', 'score': 0.54},
 {'box_2d': [1442, 575, 19, 38], 'tooth_id': '23', 'label': 'Filling', 'score': 0.34}
]

Bone loss detection (total: 1):
[
 {'box_2d': [1244, 530, 548, 92], 'label': 'Bone loss', 'site': 'lower left','score': 0.55}
]

Mandibular canal visibility (total: 2):
[
 {'box_2d': [1833, 796, 465, 531], 'label': 'Mandibular canal', 'score': 0.94},
 {'box_2d': [649, 694, 412, 562], 'label': 'Mandibular canal', 'score': 0.9}
]
"""

example_output = """
This is a panoramic dental X-ray image, which provides a broad view of the entire mouth, including the teeth, jawbones, sinuses, and other structures. Below is a detailed analysis of the image:

### Teeth-Specific Observations

1. General Condition:
    - 30 teeth visualized with findings suggestive of clear anatomical definition
    - 3 wisdom teeth detected:
        - #18: Erupted (suspected, recommend clinical review)
        - #28: Impacted (**) (suspected, recommend clinical review)
        - #38: Erupted (suspected, recommend clinical review)
2. Pathological Findings:
    - #18: sign of initial caries (*)
    - #38: Suspected deep caries (**)
    - #35: Imaging features sign of periapical granuloma (**)
3. Historical Interventions:
    - #16,46,15,25,24,23, 38: sign of fillings
    - #35: Imaging features sign of crown-implant complex (**)
    - #25: signs of root canal treatment with post-core restoration

### Jaw-Specific Observations

1. Bone Architecture:
    - Suspected mild bone loss in the lower left mandibular quadrant (*)
2. Neurovascular Structures:
    - Imaging features signs of bilateral mandibular canals

### Clinical Summary & Recommendations

1. Priority Concerns:
    - Symptomatic periapical lesion at #35 requires urgent endodontic evaluation
    - Suspected impacted #28 warrants surgical consultation for extraction planning
    - Progressive caries in #38 needs immediate intervention
2. Preventive Measures:
    - Monitor suspected initial caries at #18 with radiographic follow-up
    - Periodontal evaluation recommended for lower left quadrant bone loss (uncertain severity)
3. Follow-up Protocol:
    - 6-month recall for caries monitoring (particularly suspected areas)
    - Bitewing series recommended for interproximal caries detection

Further clinical correlation with physical examination and patient history is recommended for a comprehensive diagnosis.

--

Annotation Key:

- (**) Requires immediate intervention: Pathologies with high progression risk or critical anatomical involvement
- (*) Requires clinical attention: Early-stage findings needing monitoring or preventive management
"""