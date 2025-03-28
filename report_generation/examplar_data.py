example_input = """
This localization caption provides multi-dimensional spatial analysis of anatomical structures and pathological findings for this panoramic dental X-ray image, including:
 Teeth visibility with center points (total: 26):
[
 {'point_2d': [1290, 550], 'tooth_id': '21', 'score': 0.9},
 {'point_2d': [1272, 812], 'tooth_id': '31', 'score': 0.89},
 {'point_2d': [1352, 538], 'tooth_id': '22', 'score': 0.88},
 {'point_2d': [1156, 543], 'tooth_id': '12', 'score': 0.87},
 {'point_2d': [938, 480], 'tooth_id': '16', 'score': 0.86},
 {'point_2d': [1220, 555], 'tooth_id': '11', 'score': 0.85},
 {'point_2d': [919, 750], 'tooth_id': '46', 'score': 0.84},
 {'point_2d': [1460, 782], 'tooth_id': '35', 'score': 0.81},
 {'point_2d': [1213, 818], 'tooth_id': '41', 'score': 0.79},
 {'point_2d': [831, 450], 'tooth_id': '17', 'score': 0.75},
 {'point_2d': [805, 710], 'tooth_id': '47', 'score': 0.73},
 {'point_2d': [1014, 792], 'tooth_id': '45', 'score': 0.7},
 {'point_2d': [1390, 813], 'tooth_id': '34', 'score': 0.69},
 {'point_2d': [1660, 462], 'tooth_id': '26', 'score': 0.68},
 {'point_2d': [1154, 822], 'tooth_id': '42', 'score': 0.67},
 {'point_2d': [1098, 512], 'tooth_id': '13', 'score': 0.67},
 {'point_2d': [1076, 818], 'tooth_id': '44', 'score': 0.65},
 {'point_2d': [730, 446], 'tooth_id': '18', 'score': 0.65},
 {'point_2d': [1782, 648], 'tooth_id': '38', 'score': 0.59},
 {'point_2d': [1748, 430], 'tooth_id': '28', 'score': 0.56},
 {'point_2d': [1330, 818], 'tooth_id': '33', 'score': 0.56},
 {'point_2d': [1019, 501], 'tooth_id': '15', 'score': 0.5},
 {'point_2d': [1678, 707], 'tooth_id': '36', 'score': 0.5},
 {'point_2d': [1564, 480], 'tooth_id': '25', 'score': 0.39},
 {'point_2d': [1481, 470], 'tooth_id': '24', 'score': 0.35},
 {'point_2d': [1412, 520], 'tooth_id': '23', 'score': 0.3}
]

Wisdom teeth detection (total: 3):
[
 {'box_2d': [672, 332, 787, 561], 'tooth_id': '18', 'is_impacted': false, 'score': 0.65},
 {'box_2d': [1680, 565, 1883, 731], 'tooth_id': '38', 'is_impacted': false, 'score': 0.59},
 {'box_2d': [1684, 325, 1812, 535], 'tooth_id': '28', 'is_impacted': false, 'score': 0.56}
]

Dental caries detection (total: 2):
[
 {'box_2d': [1695, 615, 1767, 679], 'tooth_id': '38', 'label': 'Caries', 'score': 0.56},
 {'box_2d': [1667, 556, 1912, 759], 'tooth_id': '38', 'label': 'Deep caries', 'score': 0.46}
]

Periapical lesions detection (total: 1):
[
 {'box_2d': [1473, 854, 1521, 901], 'tooth_id': '35', 'label': 'Periapical lesions (Granuloma)', 'score': 0.84}
]

Historical treatments (total: 9):
[
 {'box_2d': [931, 545, 976, 571], 'tooth_id': '16', 'label': 'Filling', 'score': 0.7},
 {'box_2d': [925, 675, 958, 702], 'tooth_id': '46', 'label': 'Filling', 'score': 0.7},
 {'box_2d': [1491, 668, 1592, 748], 'tooth_id': '35', 'label': 'Crown', 'score': 0.76},
 {'box_2d': [1732, 578, 1771, 615], 'tooth_id': '38', 'label': 'Filling', 'score': 0.81},
 {'box_2d': [982, 540, 1001, 575], 'tooth_id': '15', 'label': 'Filling', 'score': 0.43},
 {'box_2d': [1548, 741, 1633, 888], 'tooth_id': '36', 'label': 'Implant', 'score': 0.81},
 {'box_2d': [1534, 527, 1580, 579], 'tooth_id': '25', 'label': 'Filling', 'score': 0.6},
 {'box_2d': [1572, 429, 1600, 528], 'tooth_id': '25', 'label': 'Root canal treatment', 'score': 0.73},
 {'box_2d': [1486, 525, 1517, 590], 'tooth_id': '24', 'label': 'Filling', 'score': 0.54}
]

Mandibular canal visibility (total: 2):
[
 {'box_2d': [1601, 531, 2066, 1062], 'label': 'Mandibular canal', 'score': 0.94},
 {'box_2d': [443, 413, 855, 975], 'label': 'Mandibular canal', 'score': 0.9}
]

Maxillary sinuses visibility (total: 2):
[
 {'box_2d': [1380, 16, 1801, 317], 'label': 'Maxillary sinus', 'score': 0.81},
 {'box_2d': [699, 15, 1084, 302], 'label': 'Maxillary sinus', 'score': 0.82}
]
"""

example_output = """
This is a panoramic dental X-ray image, which provides a broad view of the entire mouth, including the teeth, jawbones, sinuses, and other structures. Below is a detailed analysis of the image:

### Teeth-Specific Observations

1. General Condition:
    - 26 teeth visualized with findings suggestive of clear anatomical definition
    - 3 wisdom teeth detected:
        - #18: Erupted (suspected, recommend clinical review)
        - #28: Erupted (suspected, recommend clinical review)
        - #38: Erupted (suspected, recommend clinical review)
2. Pathological Findings:
    - #18: sign of caries
    - #38: Suspected deep caries
    - #35: Imaging features sign of periapical granuloma
3. Historical Interventions:
    - #15,16,23,24,25,38: sign of fillings
    - #35: Imaging features sign of dental implant with crown restoration
    - #25: signs of root canal treatment with post-core restoration

### Jaw-Specific Observations

1. Bone Architecture:
    - No apparent bone loss in the image
2. Visible Structures:
    - Imaging features signs of bilateral mandibular canals

### Clinical Summary & Recommendations

1. Priority Concerns:
    - Periapical lesion at #35 requires endodontic evaluation
    - Deep caries in #38 needs immediate intervention
2. Preventive Measures:
    - Monitor suspected caries at #18 with radiographic follow-up
3. Follow-up Protocol:
    - 6-month recall for caries monitoring (particularly suspected areas)
    - Bitewing series recommended for interproximal caries detection

Further clinical correlation with physical examination and patient history is recommended for a comprehensive diagnosis.
"""