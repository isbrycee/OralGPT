import json
import random

# question_templates = [
# "This is an intraoral video from real conservative dental treatments. Please help me generate a detailed video description.",
# "Here is an intraoral recording of genuine conservative dental treatments—please help me compose a thorough video description.",
# "This is an intraoral video from real conservative dental treatments. Could you generate a detailed description for it?",
# "This is an intraoral video from real conservative dental treatments. Please create a detailed description for it.",
# "This is an intraoral video from real conservative dental treatments. Help me to understand it accurately."
# ]


question_templates = [
    "What does a close-up 3D scan of the teeth typically show?",
    "Can you explain the details visible in a close-up 3D model of the dental structure?",
    "Could you describe what is shown in a close-up dental 3D model scan?",
    "Please describe the key elements visible in a detailed 3D model scan of the teeth.",
    "What can be observed in a 3D dental model showing a close-up of the jaw and teeth?"
]

def add_source_field_to_json(file_path):
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 检查是否是列表且每个元素是字典
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # image = item.get('image', '').split('data-train/')[-1]
                    # item['file_name'] = "MMOral-Omni/5.1_histopathological_images/" + image
                    # item['image'] = image
                    item['question'] = random.choice(question_templates)
                    # item['source'] = "UFSC-OCPap"
                    # item["Modality"] = "Histopathological images"
                    # item['split'] = "train"
                    
                    # file_name = item.get('file_name', '')
                    # aa = file_name.split('/')[0]
                    # bb = file_name.split('/')[1]
                    # file_name = f"{aa}/image/{bb}"
                    # item['file_name'] = file_name
                else:
                    print("警告：列表中存在非字典元素，已跳过")
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            
            print(f"成功在所有字典中添加 'source' 字段，并保存到原文件")
        else:
            print("错误：JSON 文件中的数据不是一个列表")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
    except json.JSONDecodeError:
        print("错误：文件内容不是有效的 JSON 格式")
    except Exception as e:
        print(f"发生未知错误: {str(e)}")

# 示例使用
if __name__ == "__main__":
    file_path = "/home/jinghao/projects/x-ray-VLM/RGB/MMOral-Omni/meta_json/8.1_3DModelScan_Captioning_all.json"
    add_source_field_to_json(file_path)