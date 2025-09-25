import pandas as pd

def capitalize_first_letters(string_list):
    """
    将列表中每个字符串的首字母和空格后的第一个字母转换为大写
    
    参数:
        string_list: 包含字符串的列表
        
    返回:
        处理后的新列表
    """
    result = []
    for s in string_list:
        # 将字符串拆分为单词列表
        words = s.split()
        # 对每个单词进行首字母大写处理
        capitalized_words = [word.capitalize() for word in words]
        # 重新组合成字符串
        result.append(" ".join(capitalized_words))
    return result

def update_category(input_file: str, output_file: str):
    # 读取 tsv 文件
    df = pd.read_csv(input_file, sep="\t", dtype=str)
    
    # 确保 category 列存在
    if "category" not in df.columns:
        raise ValueError("输入文件缺少 'category' 列")
    
    # 定义目标类别
    target_categories_for_treatment_planning = {"endodontics", "Implant Dentistry", "Periodontics"}
    
    def adjust_category_for_treatment_planning(value):
        if pd.isna(value):  # 空值处理
            return value
        categories = [c.strip() for c in value.split(",")]
        # 如果包含目标类别之一，就加 Treatment Planning
        if any(c in target_categories_for_treatment_planning for c in categories):
            if "Treatment Planning" not in categories:
                categories.append("Treatment Planning")
        categories = capitalize_first_letters(categories)
        return ",".join(categories)


    target_categories_for_pure_text_examination = {"Oral Histopathology", "Oral Mucosal Disease", "Oral & Maxillofacial Radiology"}
    
    def adjust_category_for_pure_text_examination(value):
        if pd.isna(value):  # 空值处理
            return value
        categories = [c.strip() for c in value.split(",")]
        # 如果包含目标类别之一，就加 Treatment Planning
        if any(c in target_categories_for_pure_text_examination for c in categories):
            if "Pure-text Examination" not in categories:
                categories.append("Pure-text Examination")
        categories = capitalize_first_letters(categories)
        return ",".join(categories)
    
    # 应用调整逻辑
    df["category"] = df["category"].apply(adjust_category_for_treatment_planning)
    df["category"] = df["category"].apply(adjust_category_for_pure_text_examination)
    
    
    # 保存结果到新的 tsv 文件
    df.to_csv(output_file, sep="\t", index=False)

if __name__ == "__main__":
    # 修改成你的输入 / 输出路径
    update_category("/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral.tsv", 
                    "/home/jinghao/projects/x-ray-VLM/VLMEvalKit/dataset/MMOral_new.tsv")
