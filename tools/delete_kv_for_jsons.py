import json
import os
import shutil

def remove_properties_field(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # 读取JSON文件
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 删除perporties字段（如果存在）
                if 'properties' in data:
                    del data['properties']

                if 'revised_med_report' in data:
                    if isinstance(data['revised_med_report'], str):
                        data['med_report'] = data['revised_med_report']
                    elif data['revised_med_report']['need revision'] == 'true':
                        med_report = data['Revised med report']['Revised med report']
                        data['med_report'] = med_report
                    del data['revised_med_report']

                # 写入新文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                # print(f"处理成功: {filename}")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 配置路径（修改为你的实际路径）
    input_directory = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/MM-Oral-OPG-jsons-loc-med-report"  # 原始JSON文件目录
    output_directory = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/data_0415/MM-Oral-OPG-loc-med-reports/"  # 处理后的保存目录

    remove_properties_field(input_directory, output_directory)
    print("处理完成！")
