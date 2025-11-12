import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# -----------------------------
# 设置字体为 Times New Roman
# -----------------------------
matplotlib.rcParams['font.family'] = 'Times New Roman'

# -----------------------------
# 数据定义
# -----------------------------

First_modality_number_dict = {
    "Interleaved Image-Text data": 15, # Treatment planning
    # "Plain text data": 362,
    "Intraoral Image": 1565, #  (Location and Counting) 
    "Periapical Radiograph": 539,
    "Histopathological Image": 383,
    "Cephalometric Radiograph": 300,
    "Intraoral Video": 10,
}

Second_task_number_dict = {
    "Dental treatment video comprehension": 10,
    "Treatment Planning": 15,
    "Cervical vertebral maturation (CVM) stage Prediction": 300,
    # "Examination Question": 362,
    "Abnormality Diagnosis": 2387,
    "Tooth Location and Counting": 100,
}

# -----------------------------
# 配色方案
# -----------------------------

# color_map_First_modality_blue = {
#     "Plain text data": "#A5B4FC",          
#     "Intraoral Image": "#98E4FF",         
#     "Periapical Radiograph": "#93C5FD",   
#     "Cephalometric Radiograph": "#22D3EE",
#     "Histopathological Image": "#38BDF8", 
#     "Intraoral Video": "#3B82F6",         
#     "Interleaved Image-Text data": "#BAE6FD", 
# }

# color_map_Second_task_red = {
#     "Abnormality Diagnosis": "#FCA5A5",          
#     "Treatment Planning": "#FEC6C6", 
#     "Examination Question": "#FECACA",    
#     "Cervical vertebral maturation (CVM) stage Prediction": "#F87171", 
#     "Dental treatment video comprehension": "#FB7185",  
#     "Tooth Location and Counting": "#EF4444",          
# }
# color_map_Second_task_red = {
#     "Abnormality Diagnosis": "#FCA5A5",          # 主浅红色（主色调，柔和不刺眼）
#     "Treatment Planning": "#FEC6C6", # FDB4B4         # 稍浅一点的粉红，明亮柔和
#     "Examination Question": "#FECACA",    # 更淡的粉调，增强层次
#     "Cervical vertebral maturation (CVM) stage Prediction": "#F87171", # 略偏珊瑚红，带亮度差异
#     "Dental treatment video comprehension": "#FB7185",  # 偏玫瑰粉色，微调到更饱和一点
#     "Tooth Location and Counting": "#EF4444",          # 饱和度更高的红，充当视觉重点
#     "Interleaved Image-Text data": "#FEC6C6" # 介于主色与最浅色之间的过渡色
# }


color_map_First_modality_red = {
    "Plain text data": "#FCA5A5",          
    "Intraoral Image": "#FEC6C6",         
    "Periapical Radiograph": "#F87171",   
    "Cephalometric Radiograph": "#FB7185",
    "Histopathological Image": "#EF4444", 
    "Intraoral Video": "#FECACA",         
    "Interleaved Image-Text data": "#FCA5A5", 
}

color_map_First_modality_fef9f2 = {
    "Plain text data": "#F2E5D0",          # 中浅暖米色
    "Intraoral Image": "#F9EEDC",          # 浅奶杏色
    "Periapical Radiograph": "#E8D1AD",    # 稍深的金米色
    "Cephalometric Radiograph": "#FEF4E6", # 偏深的暖沙黄
    "Histopathological Image": "#D7BF8A",  # 🟢 调浅版 深金棕 → 柔和金杏色
    "Intraoral Video": "#D8BD8B",          # 最浅层，用主色
    "Interleaved Image-Text data": "#F2E5D0", # 一致性
}


color_map_First_modality_fff5fd = {
    "Plain text data": "#DA96C8",          # 中浅粉紫（带轻微玫瑰感）
    "Intraoral Image": "#FBE7F7",          # 浅粉白（接近主色）
    "Periapical Radiograph": "#ECB9E1",    # 中粉紫（柔和亮丽）
    "Cephalometric Radiograph": "#FFF5FD", # 偏深、温柔的丁香紫
    "Histopathological Image": "#F5DAEE",  # 稍深的玫瑰紫粉（最深层）
    "Intraoral Video": "#C47DB0",          # 最浅主色（奶粉白）
    "Interleaved Image-Text data": "#F5DAEE", # 保持一致层次
}

color_map_First_modality_new = {
    "Plain text data": "#D1E7F7",      # 柔和浅蓝
    "Intraoral Image": "#E3F2FD",      # 极浅蓝
    "Periapical Radiograph": "#CCE2DC",# 淡青色
    "Cephalometric Radiograph": "#D0EBEB", # 浅蓝绿
    "Histopathological Image": "#B2C9D9",  # 蓝灰色
    "Intraoral Video": "#A8D8EA",      # 薄荷蓝
    "Interleaved Image-Text data": "#E6F3F5", # 雾感极浅蓝
}

color_map_Second_task_blue = {
    "Abnormality Diagnosis": "#93C5FD",
    "Treatment Planning": "#38BDF8", 
    "Examination Question": "#98E4FF",    
    "Cervical vertebral maturation (CVM) stage Prediction": "#98E4FF",
    "Dental treatment video comprehension": "#A5B4FC",  
    "Tooth Location and Counting": "#22D3EE", 
}

color_map_Second_task_yellow_soft = {
    "Abnormality Diagnosis": "#F6BE4F",     # 柔和亮金黄（主色，温柔不刺眼）
    "Treatment Planning": "#FFDB6E",        # 温润的柠檬黄（略亮，用于重点）
    "Examination Question": "#FFE999",      # 适中暖黄，有视觉核心
    "Cervical vertebral maturation (CVM) stage Prediction": "#D9A441",  # 稍偏橙调，平衡过渡
    "Dental treatment video comprehension": "#E9B44C",  # 稍深但温暖
    "Tooth Location and Counting": "#FFD35C",           # 压轴暖黄，保持协调
}

color_map_First_modality_fefbc7 = {
    "Plain text data": "#F3EBA8",          # 中浅暖黄（柔和金杏）
    "Intraoral Image": "#FBF4B8",          # 浅奶黄色（接近主色）
    "Periapical Radiograph": "#E9D97E",    # 偏深金黄
    "Cephalometric Radiograph": "#DCC25C", # 深一层的暖金黄
    "Histopathological Image": "#C9AC48",  # 最深层 → 柔和金棕黄（不过饱和）
    "Intraoral Video": "#FEFBC7",          # 主色本身（奶黄）
    "Interleaved Image-Text data": "#F3EBA8", # 同层次保持一致
}

def pie_stretch(
    ax,
    x,
    explode=None,
    colors=None,
    startangle=0,
    radius=1,
    wedgeprops=None,
    labels=None,
    labeldistance=1.1,
    textprops=None,
    **kwargs
):
    """
    类似于 ax.pie() 的函数，但 explode 表示外缘向外延伸（保持起始角不动）。
    """

    # 参数处理
    if wedgeprops is None:
        wedgeprops = {}
    if textprops is None:
        textprops = {}
    if explode is None:
        explode = [0] * len(x)
    if colors is None:
        colors = plt.cm.tab10(np.arange(len(x)))

    # 转为 numpy 数组
    x = np.asarray(x)
    explode = np.asarray(explode)

    # 计算各扇区角度
    total = np.sum(x)
    angles = x / total * 360.0
    angle_edges = np.cumsum([0] + list(angles)) + startangle

    # 半径参数
    width = wedgeprops.get('width', radius * 0.3) if 'width' in wedgeprops else radius * 0.3
    edgecolor = wedgeprops.get('edgecolor', 'white')
    linewidth = wedgeprops.get('linewidth', 1.0)

    # 存储结果
    wedges, texts = [], []

    for i, (theta1, theta2, color, exp) in enumerate(zip(angle_edges[:-1], angle_edges[1:], colors, explode)):
        # 内外半径：内边不变，外边扩展
        inner_radius = radius - width
        outer_radius = radius + exp * radius

        # 创建 wedge
        wedge = plt.matplotlib.patches.Wedge(
            center=(0, 0),
            r=outer_radius,
            theta1=theta1,
            theta2=theta2,
            width=outer_radius - inner_radius,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            **kwargs
        )
        ax.add_patch(wedge)
        wedges.append(wedge)

        # 添加标签
        if labels is not None:
            theta_mid = np.deg2rad((theta1 + theta2) / 2)
            label_r = outer_radius * labeldistance
            x_text, y_text = np.cos(theta_mid) * label_r, np.sin(theta_mid) * label_r
            txt = ax.text(x_text, y_text, str(labels[i]), ha='center', va='center', **textprops)
            texts.append(txt)

    # 坐标设置
    ax.set_aspect('equal')
    ax.set_xlim(-radius * (1 + max(explode) * 1.5), radius * (1 + max(explode) * 1.5))
    ax.set_ylim(-radius * (1 + max(explode) * 1.5), radius * (1 + max(explode) * 1.5))

    return wedges, texts
    
# -----------------------------
# 数据匹配和排序
# -----------------------------

def create_matched_data(inner_dict, outer_dict):
    """创建匹配的内外环数据"""
    # 找出匹配的值
    inner_values = list(inner_dict.values())
    outer_values = list(outer_dict.values())
    
    # 找出所有唯一值并按降序排序
    all_unique_values = sorted(set(inner_values + outer_values), reverse=True)
    
    inner_sorted = []
    outer_sorted = []
    inner_labels = []
    outer_labels = []
    inner_colors = []
    outer_colors = []
    
    for value in all_unique_values:
        # 处理内环数据
        for key, val in inner_dict.items():
            if val == value:
                inner_sorted.append(value)
                inner_labels.append(key)
                inner_colors.append(color_map_First_modality_fff5fd.get(key, "#aaaaaa"))
        
        # 处理外环数据
        for key, val in outer_dict.items():
            if val == value:
                outer_sorted.append(value)
                outer_labels.append(key)
                outer_colors.append(color_map_Second_task_blue.get(key, "#aaaaaa"))
    
    return (inner_sorted, inner_labels, inner_colors, 
            outer_sorted, outer_labels, outer_colors)

# 创建匹配的数据
(inner_values, inner_labels, inner_colors, 
 outer_values, outer_labels, outer_colors) = create_matched_data(
    First_modality_number_dict, Second_task_number_dict
)

index = outer_values.index(100)
# 移除 100
value = outer_values.pop(index)
# 将 100 插入到第二个位置（索引 1）
outer_values.insert(0, value)

index = outer_labels.index('Tooth Location and Counting')
# 移除 100
value = outer_labels.pop(index)
# 将 100 插入到第二个位置（索引 1）
outer_labels.insert(0, value)

# -----------------------------
# 创建外环的爆炸效果
# -----------------------------

# -----------------------------
# 创建爆炸效果 - 分别处理内外环
# -----------------------------

# 内环没有爆炸效果
explode_inner = [0] * len(inner_values)

# 创建外环的爆炸距离列表
explode_outer = [0] * len(outer_values)

# 找到外环中两个最小的值（占比很小的部分）
outer_sorted_with_indices = sorted([(value, idx) for idx, value in enumerate(outer_values)])
min_indices = [item[1] for item in outer_sorted_with_indices[:2]]  # 获取两个最小值的索引

# 为这两个最小的部分设置不同的爆炸距离
# 第一个最小值延伸0.15，第二个最小值延伸0.1
explode_outer[min_indices[0]] = 0.15  # 第一个最小值延伸更多
explode_outer[min_indices[1]] = 0.1   # 第二个最小值延伸较少

# import pdb; pdb.set_trace()
# -----------------------------
# 绘制双层圆环图
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制外环（第二任务分类）
wedges_outer, texts_outer = pie_stretch(
    ax,
    outer_values,
    colors=outer_colors,
    startangle=90,
    radius=0.9,  # 外环半径
    wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),
    explode=explode_outer  # 添加爆炸效果
)

# 绘制内环（第一模态分类）
wedges_inner, texts_inner = ax.pie(
    inner_values,
    colors=inner_colors,
    startangle=90,
    radius=0.6,  # 内环半径
    wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1),

)

# 在中心添加一个白色圆圈，形成圆环效果
centre_circle = plt.Circle((0, 0), 0.1, fc='white')
ax.add_artist(centre_circle)

# 设置坐标轴属性
ax.axis('equal')

# -----------------------------
# 打印统计信息
# -----------------------------
print("内环各模态数量占比：")
inner_total = sum(inner_values)
for label, value in zip(inner_labels, inner_values):
    ratio = value / inner_total * 100
    print(f"{label:45s}: {value:6d} ({ratio:6.2f}%)")

print("\n外环各任务数量占比：")
outer_total = sum(outer_values)
for label, value in zip(outer_labels, outer_values):
    ratio = value / outer_total * 100
    print(f"{label:45s}: {value:6d} ({ratio:6.2f}%)")

# -----------------------------
# 添加标题
# -----------------------------
# plt.title('Dual-layer Donut Chart: Modality vs Task Distribution\n',
#           fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('dual_layer_donut_chart_v1.png', bbox_inches='tight', pad_inches=0.01, dpi=300)