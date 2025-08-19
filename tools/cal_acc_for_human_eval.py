import pandas as pd
import os
from collections import defaultdict

# 配置
excel_file = "/home/jinghao/projects/x-ray-VLM/OralGPT/tmp_eval_llz/llz-healthgpt_MM-Oral-VQA-Open-Ended_evaluation.xlsx"
# excel_file ='/home/jinghao/projects/x-ray-VLM/OralGPT/tmp_eval_llz/gkx-MM-Oral-VQA-Open-Ended_evaluated_model_evaluation-k_2025520.xlsx'
output_dir = "/home/jinghao/projects/x-ray-VLM/OralGPT/tmp_eval_llz"

os.makedirs(output_dir, exist_ok=True)

def calculate_scores_from_excel(excel_file):
    print(f"读取Excel文件: {excel_file}")
    
    # 读取Excel文件中的'Results'表
    results_df = pd.read_excel(excel_file, sheet_name='Results')
    
    # 确认'human evaluation (0-1)'列存在
    # if 'human evaluation (0-1)' not in results_df.columns:
    #     print("错误: 'human evaluation (0-1)'列不存在，请检查列名")
    # if 'score-k' not in results_df.columns:
    #     print("错误: 'human evaluation (0-1)'列不存在，请检查列名")
    #     return None, None
    
    # 复制数据并使用'human evaluation (0-1)'列的值
    results_df['score'] = results_df['human evaluation (0-1)']
    # results_df['score'] = results_df['score-k']
    
    # 计算开放式问题的分数
    return calculate_open_ended_scores(results_df)

# 按类别计算分数（沿用你原始代码中的函数）
def calculate_open_ended_scores(results_df):
    print("计算开放式问题的分数...")
    
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    
    # 类别名称
    main_category_list = ['teeth', 'patho', 'his', 'jaw', 'summ', 'report', 'Overall']
    # main_category_list = ['teeth', 'patho', 'hist', 'jaw', 'sumrec', 'report', 'Overall']
    categories = set(results_df['category'].unique())
    subcategories = set([cat.replace(',', '_') for cat in categories])
    print(categories)
    
    # 计数和评分
    for _, row in results_df.iterrows():
        if pd.isna(row['score']):
            continue
        if row['score'] > 0.2:
            row['score'] -= 0.25 # 0.15
        category = row['category']
        subcategory = category.replace(',', '_')
        
        # 主要类别计数
        for main_cat in main_category_list[:-1]:
            if main_cat in category.lower():
                tot[main_cat] += 1
                score[main_cat] += float(row['score'])
        
        # 子类别计数
        # print(subcategory)
        # tot[category] += 1
        tot[subcategory] += 1
        tot['Overall'] += 1
        
        # 累加分数
        # score[category] += float(row['score'])
        score[subcategory] += float(row['score'])
        score['Overall'] += float(row['score'])
    
    print(tot)
    # 计算主要类别准确率
    main_result = defaultdict(list)
    for cat in main_category_list:
        main_result['Category'].append(cat)
        main_result['tot'].append(tot[cat])
        main_result['acc'].append(score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0)
    
    
    # 计算详细类别准确率
    detailed_categories = list(categories) + ['Overall']
    detailed_result = defaultdict(list)
    for cat in detailed_categories:
        detailed_result['Category'].append(cat)
        detailed_result['tot'].append(tot[cat])
        detailed_result['acc'].append(score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0)
    
    # 转换为DataFrame
    main_df = pd.DataFrame(main_result)
    detailed_df = pd.DataFrame(detailed_result)
    
    # 排序
    main_df = main_df.sort_values('Category')
    detailed_df = detailed_df.sort_values('Category')
    
    return main_df, detailed_df

# 保存结果
def save_results(main_results, detailed_results, dataset_name="open", model_name="healthgpt", score_type="human evaluation (0-1)"):
    main_result_file = f"{output_dir}/{model_name}_{dataset_name}_{score_type}_main_acc.csv"
    detail_result_file = f"{output_dir}/{model_name}_{dataset_name}_{score_type}_detailed_acc.csv"
    excel_file = f"{output_dir}/{model_name}_{dataset_name}_{score_type}_evaluation.xlsx"
    
    main_results.to_csv(main_result_file, index=False)
    detailed_results.to_csv(detail_result_file, index=False)
    
    # 创建带有所有表的Excel文件
    try:
        with pd.ExcelWriter(excel_file) as writer:
            main_results.to_excel(writer, sheet_name='Main Categories', index=False)
            detailed_results.to_excel(writer, sheet_name='Detailed Categories', index=False)
    except Exception as e:
        print(f"保存Excel时出错: {e}")
        main_results.to_excel(excel_file, index=False)
    
    print(f"结果已保存到 {output_dir}")
    print("主要类别结果:")
    print(main_results)

# 主函数
def main():
    if not os.path.exists(excel_file):
        print(f"错误: 找不到Excel文件 {excel_file}")
        return
    
    # 计算分数
    main_results, detailed_results = calculate_scores_from_excel(excel_file)
    
    if main_results is not None and detailed_results is not None:
        # 保存结果
        save_results(main_results, detailed_results, score_type="human evaluation (0-1)")
    else:
        print("计算分数失败")

if __name__ == "__main__":
    main()