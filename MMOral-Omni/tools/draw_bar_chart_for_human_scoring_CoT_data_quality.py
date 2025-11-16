import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

# 示例数据
scores_person1 = [4.20, 4.30, 4.42, 4.47, 4.39, 4.40, 4.51] # llz
scores_person2 = [4.29, 4.44, 4.51, 4.53, 4.62, 4.63, 4.66] # zwk
categories = ['Holistic Quality', 'Explainability', 'Caption Acceptance', 'Rationality of Thinking', 'Correctness of Answer', 'Completeness of Answer', 'Relevance and Clarity']

# 设置柱状图的宽度
bar_width = 0.35
x = np.arange(len(categories))

# 创建柱状图
fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - bar_width/2, scores_person1, bar_width, label='Dentist 1', 
                color='#f5dbf0', edgecolor='black', linewidth=1.5)  # 淡紫色
bars2 = ax.bar(x + bar_width/2, scores_person2, bar_width, label='Dentist 2', 
                color='#ecb8e1', edgecolor='black', linewidth=1.5)  # 更深的淡紫色

# 添加一些美观的元素
# ax.set_xlabel('Categories', fontsize=14)
ax.set_ylabel('Scores', fontsize=18, fontweight='bold')
# ax.set_title('Scores Comparison', fontsize=16)
ax.set_xticks(x)
# ax.set_xticklabels(categories)
ax.set_xticklabels(categories, rotation=12, fontsize=14, ha='center', fontweight='bold')  # 旋转标签
ax.set_yticklabels(range(0,5), fontsize=15, fontweight='bold')  # 旋转标签

ax.legend(fontsize=14)

# 加粗坐标轴线条
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                ha='center', va='bottom', fontsize=14)

add_value_labels(bars1)
add_value_labels(bars2)

# 显示网格
ax.yaxis.grid(True, linestyle='--', color='gray')

# 美化布局
plt.tight_layout()

# 保存图表为文件
plt.savefig('human_scoring_CoT_data_quality.png', dpi=300)

# # 显示图表
# plt.show()
