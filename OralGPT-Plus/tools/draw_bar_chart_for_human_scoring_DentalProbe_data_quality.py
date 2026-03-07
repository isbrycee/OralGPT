import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

# 示例数据
scores_person1 = [
    3.775147929,
    4.236686391,
    3.911242604,
    3.769230769,
    3.73964497,
    3.810650888
] # bjh

categories = [
    "Overall Quality Assessment",
    "Region Zoom-in Acceptance",
    "Logical consistency",
    "Correctness of Answer",
    "Completeness of Answer",
    "Relevance and Clarity"
]

# 设置柱状图的宽度
bar_width = 0.5
x = np.arange(len(categories))

# 创建柱状图
fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x, scores_person1, bar_width, label='Dentist Evaluation', 
                color='#f5dbf0', edgecolor='black', linewidth=1.5)  # 淡紫色

# 添加一些美观的元素
ax.set_ylabel('Scores', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_yticks(np.linspace(0, 5, 6))  # 0到5之间的6个刻度
ax.set_xticklabels(categories, rotation=12, fontsize=14, ha='center', fontweight='bold')  # 旋转标签
ax.set_yticklabels(np.linspace(0, 5, 6), fontsize=15, fontweight='bold')  # 设置y轴标签

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

# 显示网格
ax.yaxis.grid(True, linestyle='--', color='gray')

# 美化布局
plt.tight_layout()

# 保存图表为文件
plt.savefig('human_scoring_CoT_data_quality_DentalProbe.png', dpi=300)
