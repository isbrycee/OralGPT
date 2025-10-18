import pandas as pd

def filter_excel(file_path, new_file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 删除 category 列中包含 'TE,' 的行
    df_filtered = df[~df['category'].str.contains('TE,', na=False)]
    
    # 保存新的Excel文件
    df_filtered.to_excel(new_file_path, index=False)
    
    # 打印新Excel文件的行数
    print(f"新的Excel文件行数: {len(df_filtered)}")
    
    return len(df_filtered)

# 示例调用
filter_excel('/home/jinghao/projects/x-ray-VLM/OralGPT/MMOral-Omni-Bench-Eval/gpt-5-mini/T20251018_Gc2d011ff/gpt-5-mini_MMOral_OMNI_gpt-5-mini.xlsx', 
             '/home/jinghao/projects/x-ray-VLM/OralGPT/MMOral-Omni-Bench-Eval/gpt-5-mini/T20251018_Gc2d011ff/gpt-5-mini_MMOral_OMNI_gpt-5-mini-woTE.xlsx')

