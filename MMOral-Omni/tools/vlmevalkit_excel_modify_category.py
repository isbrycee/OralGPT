import pandas as pd

def map_category_in_excel(input_file, output_file, mapping_dict):
    # 读取Excel文件
    df = pd.read_excel(input_file)
    
    # 检查是否存在'category'列
    if 'category' not in df.columns:
        raise ValueError("输入文件中没有 'category' 列")
    
    # 根据映射字典map替换'category'列内容，不在字典中的保持原值
    df['category'] = df['category'].map(mapping_dict).fillna(df['category'])
    
    # 保存结果到新的Excel文件
    df.to_excel(output_file, index=False)
    print(f"映射完成，文件已保存到 {output_file}")

# 使用示例
mapping = {
'Histo,Oral Submucous Fibrosis': 'PI,Oral Submucous Fibrosis', 
'Histo,Leukoplakia with Dysplasia': 'PI,Leukoplakia with Dysplasia', 
'Histo,Leukoplakia without Dysplasia': 'PI,Leukoplakia without Dysplasia',
'Histo,Normality': 'PI,Normality',
'Histo,Oral Squamous Cell Carcinoma': 'PI,Oral Squamous Cell Carcinoma',

'PA,Root Canal Treatment': 'PA,Root Canal Treatment',
'PA,Bone Loss': 'PA,Bone Loss',
'PA,Restoration': 'PA,Restoration',
'PA,Crown': 'PA,Crown',
'PA,Pulpitis': 'PA,Pulpitis',
'PA,Periodontitis': 'PA,Periodontitis',
'PA,Mixed Dentition': 'PA,Mixed Dentition',
'PA,Apical Periodontitis': 'PA,Apical Periodontitis',
'PA,Caries': 'PA,Caries',
'PA,Impacted Tooth': 'PA,Impacted Tooth',

'II_loc': 'II_Loc',

'II_I_diag,Gingivitis': 'II_Dx-I,Gingivitis', 
'II_I_diag,Orthodontics': 'II_Dx-I,Orthodontics', 
'II_I_diag,Calculus': 'II_Dx-I,Calculus',
'II_I_diag,Defective Dentition': 'II_Dx-I,Defective Dentition',
'II_I_diag,Normality': 'II_Dx-I,Normality',
'II_I_diag,Tooth Discoloration': 'II_Dx-I,Tooth Discoloration',
'II_I_diag,Ulcer': 'II_Dx-I,Ulcer',
'II_I_diag,Caries': 'II_Dx-I,Caries',
'II_I_diag,Cancer': 'II_Dx-I,Cancer',

'II_R_diag,Fenestration and Dehiscence': 'II_Dx-R,Fenestration and Dehiscence',
'II_R_diag,Malocclusion Issues Assessment': 'II_Dx-R,Malocclusion Issues Assessment',
'II_R_diag,Caries': 'II_Dx-R,Caries',
'II_R_diag,Gingivitis': 'II_Dx-R,Gingivitis',

'Intraoral Video': 'IV',

'Endodontics,Treatment Planning': 'TP,Endodontics',
'Implant Dentistry,Treatment Planning': 'TP,Implant Dentistry',
'Periodontics,Treatment Planning': 'TP,Periodontics',

'Oral Mucosal Disease,Pure-text Examination': 'TE,Oral Mucosal Disease', 
'Oral & Maxillofacial Radiology,Pure-text Examination': 'TE,Oral & Maxillofacial Radiology', 
'Oral Histopathology,Pure-text Examination': 'TE,Oral Histopathology',

'Cepha': 'CE',
}
map_category_in_excel('/home/jinghao/projects/x-ray-VLM/OralGPT/MMOral-Omni-Bench-Eval/OralGPT-Omni_qwen2_5vl-7b_baseline/T20251014_Gfb8c54fa/OralGPT-Omni_qwen2_5vl-7b_baseline_MMOral_gpt-5-mini.xlsx', 
                      '/home/jinghao/projects/x-ray-VLM/OralGPT/MMOral-Omni-Bench-Eval/OralGPT-Omni_qwen2_5vl-7b_baseline/T20251014_Gfb8c54fa/OralGPT-Omni_qwen2_5vl-7b_baseline_MMOral_gpt-5-mini.xlsx', 
                      mapping)

