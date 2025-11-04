import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

system_prompt = """You are a senior dental radiologist. Your task is to carefully review the provided panoramic x-ray image and identify any patient positioning errors present. For each error detected, explain the observable radiographic findings in the image that support your conclusion. Your explanation should demonstrate clear radiological reasoning.

There are 19 predefined potential positioning errors:
1: Chin Tipped High
2: Chin Tipped Low
3: Head Tilted to the Left
4: Head Tilted to the Right
5: Head Turned toward the Left
6: Head Turned toward the Right
7: Patient Positioned Backward
8: Patient Positioned Forward
9: Slumped Neck Position
10: Condyles Cut Off the Image
11: Teeth Not Disoccluded
12: Tongue Not Against the Roof of the Mouth
13: Metal Artifact
14: Patient’s Chin Not Against the Chin Rest
15: Lead Apron Artifact
16: Midline Shifted to the Left
17: Midline Shifted to the Right
18: Patient Movement
19: No Obvious Error

You MUST strictly follow the output format below. Ensure all opening and closing tags are written exactly as specified, without omission, modification, or misspelling:
<Think> Provide a concise reasoning process describing how you identified the errors based on radiographic features. </Think>
<Errors> List the detected positioning errors by their number and description. </Errors>
<Correction> Provide precise instructions on how to correct the errors during image acquisition. </Correction>"""

fix_response_for_images_no_errors = """
<Think>The panoramic radiograph demonstrates proper patient positioning. The occlusal plane is level and symmetrical. The orbits and mandibular rami appear evenly aligned with no evidence of tilt or rotation. The anterior and posterior teeth are proportionate in width, indicating correct anteroposterior positioning. The condyles are fully captured within the image. The tongue is placed against the palate, eliminating any radiolucent shadow in the maxillary region. No artifacts are present. Overall, this image does not show any positioning errors.</Think>
<Errors>19: No Obvious Error</Errors>
<Correction>No corrective action is required. Patient positioning was performed correctly.</Correction>"""

# PREDEFINED_CLASSES = [
#     "Chin Tipped High",
#     "Chin Tipped Low",
#     "Head Tilted to the Left",
#     "Head Tilted to the Right",
#     "Head Turned toward the Left",
#     "Head Turned toward the Right",
#     "Patient Positioned Backward",
#     "Patient Positioned Forward",
#     "Slumped Neck Position",
#     "Condyles Cut Off the Image",
#     "Teeth Not Disoccluded",
#     "Tongue Not Against the Roof of the Mouth",
#     "Metal Artifact",
#     "Patient’s Chin Not Against the Chin Rest",
#     "Lead Apron Artifact",
#     "Midline Shifted to the Left",
#     "Midline Shifted to the Right",
#     "Patient Movement",
#     "No Obvious Error"
# ]
# PREDEFINED_CLASSES_ID_map = {element: index for index, element in enumerate(PREDEFINED_CLASSES)}

# 读取 Excel 文件
file_path = "/home/jinghao/projects/positioning-error/Labeled-data.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path, header=None)

# 处理第一行，存储为字典
first_row = df.iloc[0]
header_dict = {}
for cell in first_row:
    if isinstance(cell, str) and ':' in cell:
        num, category = cell.split(':', 1)
        header_dict[int(num.strip())] = category.strip()

# 处理第二行表头并只取前四列，从第三行开始为数据
df_data = pd.read_excel(file_path, header=1, usecols=[0, 1, 2, 3])  # header=1表示从第二行作为表头，usecols选择前四列
df_data = df_data.dropna(how='all')
# 组织数据为合适的数据结构
data_list = []
for _, row in df_data.iterrows():
    data_entry = {
        "Image Number": row["Image Number"],
        "Image Name": row["Image Name"],
        "Description": row["Description"],
        "Categories": row["Categories"]
    }
    if 'error' in str(data_entry["Categories"]):
        print('here')
        data_entry["Categories"] = data_entry["Categories"].replace("error", "")
    data_list.append(data_entry)

# 输出结果
print("第一行字典:", header_dict)
# print("数据列表:", data_list)

def ensure_ends_with_dot(input_string):
    """
    检查字符串是否以 '.' 结尾，如果没有则加上。
    
    :param input_string: str，输入的字符串
    :return: str，确保以 '.' 结尾的字符串
    """
    if not input_string.endswith('.'):
        return input_string + '.'
    return input_string

# 定义函数处理 Description 字段
def process_description(description, error_categories):
    # 分割 Description 字段
    error_part = ""
    diagnosed_part = ""
    correction_part = ""
    
    # 查找字段的位置
    if "Error:" in description:
        error_part = description.split("Error:")[1].split("Diagnosed:")[0].strip() if "Diagnosed:" in description else description.split("Error:")[1].strip()
    elif "Errors:" in description:
        error_part = description.split("Errors:")[1].split("Diagnosed:")[0].strip() if "Diagnosed:" in description else description.split("Errors:")[1].strip()

    if "Diagnosed:" in description:
        diagnosed_part = description.split("Diagnosed:")[1].split("Correction:")[0].strip() if "Correction:" in description else description.split("Diagnosed:")[1].strip()
    elif "Diagnosis:" in description:
        diagnosed_part = description.split("Diagnosis:")[1].split("Correction:")[0].strip() if "Correction:" in description else description.split("Diagnosis:")[1].strip()

    if "Correction:" in description:
        correction_part = description.split("Correction:")[1].strip()
    elif "Corrected:" in description:
        correction_part = description.split("Corrected:")[1].strip()
    elif "Correct:" in description:
        correction_part = description.split("Correct:")[1].strip()
    elif "Corrections:" in description:
        correction_part = description.split("Corrections:")[1].strip()
    else:
        correction_part = None
    
    diagnosed_part = ensure_ends_with_dot(diagnosed_part)
    error_part = ensure_ends_with_dot(error_part)
    if correction_part:
        correction_part = ensure_ends_with_dot(correction_part)

    error_category_part = []

    if isinstance(error_categories, int):
        error_category_part.append(str(error_categories) + ": " + header_dict[error_categories])
    else:
        for category_id in error_categories.split(','):
            if len(category_id.strip()) > 0:
                error_category_part.append(str(category_id) + ": " + header_dict[int(category_id.strip())])

    error_category_part_str = ", ".join(error_category_part)

    print(error_category_part_str)

    # 构造 Think, Error, Correction 部分
    think_content = f"<Think>{diagnosed_part}\n{error_part}</Think>"
    error_content = f"<Errors>{error_category_part_str}</Errors>"
    if correction_part:
        correction_content = f"<Correction>{correction_part}</Correction>"
    else:
        correction_content = ''
    
    # 返回结果
    return f"{think_content}\n{error_content}\n{correction_content}"

# 构造最终 JSON 数据
final_data = []
for row in data_list:
    # 提取 Description 并处理
    description = row["Description"]
    error_categories = row["Categories"]
    image_name = 'images_errors/' + row["Image Name"] + '.jpg'
    if not os.path.exists(os.path.join("/home/jinghao/projects/positioning-error/", image_name)):
        print(image_name)
        continue
    
    processed_content = process_description(description, error_categories)
    
    # 构造单条数据
    entry = {
        "messages": [
            {
                "content": "<image>Please analyze the panoramic x-ray image for positioning errors, according to the predefined 19 error categories. Return your findings using the required <Think>, <Errors>, and <Correction> format.",
                "role": "user"
            },
            {
                "content": processed_content,
                "role": "assistant"
            }
        ],
        "images": ['images_errors/' + row["Image Name"] + '.jpg'],
        "system": system_prompt
    }
    final_data.append(entry)

# 图像文件夹路径
image_folder = "/home/jinghao/projects/positioning-error/images_no_errors/"  # 替换为你的图像文件夹路径
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 构造与 final_data 相同格式的图像数据
image_no_errors_data = []
for img in image_files:
    entry = {
        "messages": [
            {
                "content": "<image>Please analyze the panoramic x-ray image for positioning errors, according to the predefined 19 error categories. Return your findings using the required <Think>, <Errors>, and <Correction> format.",
                "role": "user"
            },
            {
                "content": fix_response_for_images_no_errors,
                "role": "assistant"
            }
        ],
        "images": [f'images_no_errors/{img}'],
        "system": system_prompt
    }
    image_no_errors_data.append(entry)


# 按相同比例划分
train_final_data, test_final_data = train_test_split(final_data, test_size=0.1, random_state=42)
train_image_no_errors, test_image_no_errors = train_test_split(image_no_errors_data, test_size=0.1, random_state=42)

# 合并训练集和测试集
train_combined = train_final_data + train_image_no_errors
test_combined = test_final_data + test_image_no_errors

# 输出训练集和测试集的长度
print(f"训练集长度: {len(train_combined)}")
print(f"测试集长度: {len(test_combined)}")

# 保存为 JSON 文件
output_dir = "/home/jinghao/projects/positioning-error"  # 替换为你的输出路径
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "OPG_Positioning_Error_train.json"), "w", encoding="utf-8") as f:
    json.dump(train_combined, f, indent=4, ensure_ascii=False)

with open(os.path.join(output_dir, "OPG_Positioning_Error_test.json"), "w", encoding="utf-8") as f:
    json.dump(test_combined, f, indent=4, ensure_ascii=False)

print("训练集和测试集已成功划分并保存为 JSON 文件！")
