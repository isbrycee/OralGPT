# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np

# # -----------------------------
# # 设置字体为 Times New Roman
# # -----------------------------
# matplotlib.rcParams['font.family'] = 'Times New Roman'

# # -----------------------------
# # 数据定义
# # -----------------------------

# First_modality_number_dict = {
#     "Interleaved Image-Text data": 15, # Treatment planning
#     "Plain text data": 362,
#     "Intraoral Image": 1565, #  (Location and Counting) 
#     # "Intraoral Image (Image-level Analysis)": 723,
#     # "Intraoral Image (Region-level Analysis)": 742,
#     "Periapical Radiograph": 539,
#     "Histopathological Image": 383,
#     "Cephalometric Radiograph": 300,
#     "Intraoral Video": 10,
# }

# Second_task_number_dict = {
#     "Dental treatment video comprehension": 10,
#     "Treatment Planning": 15,
#     "Cervical vertebral maturation (CVM) stage Prediction": 300,
#     "Examination Question": 362,
#     "Abnormality Diagnosis": 2387,
#     "Tooth Location and Counting": 100,
# }

# # subject; abnormalities;
# Third_abnormality_number_dict = {
#     # Interleaved Image-Text data; Treatment planning
#     # 'Endodontics': 5, 
#     # 'Implant Dentistry': 5, 
#     # 'Periodontics': 5, 

#     # Examination Question; Plain text
#     # 'Oral Histopathology': 110, 
#     # 'Oral Mucosal Disease': 142, 
#     # 'Oral & Maxillofacial Radiology': 110, 

#     # II_I_diag
#     'Orthodontics': 70, 
#     'Cancer': 273, 
#     'Gingivitis': 135, # 68 + 67 
#     'Defective Dentition': 20, 
#     'Normality': 348, # 100 + 248
#     'Tooth Discoloration': 27, 
#     'Ulcer': 39, 
#     'Caries': 543, # 74 + 353 + 116
#     'Calculus': 52, 

#     # II_R_diag
#     # 'Caries': 353, 
#     'Fenestration and Dehiscence': 22, 
#     # 'Gingivitis': 67, 
#     'Malocclusion Issues Assessment': 300,

#     # PA
#     'Impacted Tooth': 101, 
#     'Pulpitis': 44, 
#     # 'Caries': 116, 
#     'Periodontitis': 68, 
#     'Apical Periodontitis': 62, 
#     'Mixed Dentition': 50, 
#     'Bone Loss': 40, 
#     'Root Canal Treatment': 11,
#     'Crown': 25, 
#     'Restoration': 22,

#     # Histo
#     # 'Normality': 248,
#     'Leukoplakia with Dysplasia': 15, 
#     'Oral Squamous Cell Carcinoma': 76, 
#     'Leukoplakia without Dysplasia': 9, 
#     'Oral Submucous Fibrosis': 35, 

#     # "Cephalometric Radiograph": 300,
#     # "Intraoral Image (Location and Counting)": 100,
#     # 'Intraoral Video': 10, 
# }

# # {'Endodontics,Treatment Planning': 5, 'Implant Dentistry,Treatment Planning': 5, 'Periodontics,Treatment Planning': 5, 'Oral Histopathology,Pure-text Examination': 110, 'Oral Mucosal Disease,Pure-text Examination': 142, 'Oral & Maxillofacial Radiology,Pure-text Examination': 110, 'II_loc': 100, 'Cepha': 300, 'II_I_diag,Orthodontics': 70, 'II_I_diag,Cancer': 273, 'II_I_diag,Gingivitis': 68, 'II_I_diag,Defective Dentition': 20, 'II_I_diag,Normality': 100, 'II_I_diag,Tooth Discoloration': 27, 'II_I_diag,Ulcer': 39, 'II_I_diag,Caries': 74, 'II_I_diag,Calculus': 52, 'PA,Impacted Tooth': 101, 'PA,Pulpitis': 44, 'PA,Caries': 116, 'PA,Periodontitis': 68, 'PA,Apical Periodontitis': 62, 'PA,Mixed Dentition': 50, 'PA,Bone Loss': 40, 'PA,Root Canal Treatment': 11, 'PA,Crown': 25, 'PA,Restoration': 22, 'Histo,Normality': 248, 'Histo,Leukoplakia with Dysplasia': 15, 'Histo,Oral Squamous Cell Carcinoma': 76, 'Histo,Leukoplakia without Dysplasia': 9, 'Histo,Oral Submucous Fibrosis': 35, 'Intraoral Video': 10, 'II_R_diag,Caries': 353, 'II_R_diag,Fenestration and Dehiscence': 22, 'II_R_diag,Gingivitis': 67, 'II_R_diag,Malocclusion Issues Assessment': 300}

# # -----------------------------
# # 配色方案
# # -----------------------------
# # 优化的配色方案 - 融入蓝色主题
# # -----------------------------
# # color_map = {
# #     "Plain text data": "#687EFF",        # 您的深蓝色
# #     "Figures from Textbooks": "#80B3FF",    # 您的中蓝色
# #     "Intraoral Image": "#98E4FF",   # 您的浅蓝色 - 三个子任务用相同颜色
# #     "Periapical Radiograph": "#FF6B6B",     # 暖红色，与蓝色形成对比
# #     "Panoramic Radiograph": "#FFD166",      # 暖黄色
# #     "Cephalometric Radiograph": "#06D6A0",  # 青绿色
# #     "Histopathological Image": "#B6FFFA",   # 您的极浅蓝色
# #     "Intraoral Video": "#C8B6FF",           # 淡紫色
# #     "Interleaved Image-Text data": "#FFA69E", # 珊瑚粉色
# #     "3D Model Scan": "#A0C4FF",             # 另一种蓝色调
# # }

# color_map_First_modality_blue = {
#     "Plain text data": "#A5B4FC",          # 主蓝色 (清晰主色)
#     "Intraoral Image": "#98E4FF",         # 浅蓝色
#     "Periapical Radiograph": "#93C5FD",   # 更浅一点的蓝，用于缓和区域
#     "Cephalometric Radiograph": "#22D3EE",# 青蓝色，带点绿调用于亮点区分
#     "Histopathological Image": "#38BDF8", # 极浅蓝，适合作为柔和色块
#     "Intraoral Video": "#3B82F6",          # 带一点紫调的浅蓝，形成微妙变化
#     "Interleaved Image-Text data": "#BAE6FD", # 海蓝色中间调，用于强调
# }

color_map_Second_task_red = {
    "Abnormality Diagnosis": "#FCA5A5",          # 主浅红色（主色调，柔和不刺眼）
    "Treatment Planning": "#FEC6C6", # FDB4B4         # 稍浅一点的粉红，明亮柔和
    "Examination Question": "#FECACA",    # 更淡的粉调，增强层次
    "Cervical vertebral maturation (CVM) stage Prediction": "#F87171", # 略偏珊瑚红，带亮度差异
    "Dental treatment video comprehension": "#FB7185",  # 偏玫瑰粉色，微调到更饱和一点
    "Tooth Location and Counting": "#EF4444",          # 饱和度更高的红，充当视觉重点
    "Interleaved Image-Text data": "#FEC6C6" # 介于主色与最浅色之间的过渡色
}

# # -----------------------------
# # 数据准备
# # -----------------------------
# labels = []
# values = []
# colors = []

# for key, val in Second_task_number_dict.items():
#     labels.append(key)
#     values.append(val)
#     colors.append(color_map_Second_task_red.get(key, "#aaaaaa"))

# # -----------------------------
# # 绘制饼图
# # -----------------------------
# fig, ax = plt.subplots(figsize=(8, 8))

# wedges, _ = ax.pie(
#     values,
#     colors=colors,
#     startangle=90,
#     wedgeprops=dict(width=0.6, edgecolor='white')  # 中心空白形成环形
# )

# # 计算比例并在 >1% 的部分添加文字
# total = sum(values)
# # for i, wedge in enumerate(wedges):
# #     ratio = 100 * values[i] / total
# #     if ratio > 0.99:
# #         # 计算角度和中心点
# #         theta = (wedge.theta2 + wedge.theta1) / 2.0
# #         x = 0.7 * np.cos(np.deg2rad(theta))
# #         y = 0.7 * np.sin(np.deg2rad(theta))
# #         ax.text(x, y, f"{ratio:.1f}%", ha='center', va='center',
# #                 fontsize=18, fontweight='bold', fontfamily='Times New Roman', color='black')

# ax.axis('equal')
# # plt.title("Modality Distribution (Ring Chart)", fontsize=14, fontfamily='Times New Roman')

# # -----------------------------
# # 打印每个模态数量占比
# # -----------------------------
# print("各模态数量占比：")
# for label, value in zip(labels, values):
#     ratio = value / total * 100
#     print(f"{label:45s}: {value:6d} ({ratio:6.2f}%)")

# plt.tight_layout()
# plt.savefig('output_benchmark.png', bbox_inches='tight', pad_inches=0.02, dpi=300)  # 保存图像到当前目录下的 output.png 文件

# # plt.show()



######
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
    "Plain text data": 362,
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
    "Examination Question": 362,
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

color_map_First_modality_red = {
    "Plain text data": "#FCA5A5",          
    "Intraoral Image": "#FEC6C6",         
    "Periapical Radiograph": "#F87171",   
    "Cephalometric Radiograph": "#FB7185",
    "Histopathological Image": "#EF4444", 
    "Intraoral Video": "#FECACA",         
    "Interleaved Image-Text data": "#FCA5A5", 
}

color_map_Second_task_blue = {
    "Abnormality Diagnosis": "#93C5FD",          
    "Treatment Planning": "#38BDF8", 
    "Examination Question": "#98E4FF",    
    "Cervical vertebral maturation (CVM) stage Prediction": "#22D3EE", 
    "Dental treatment video comprehension": "#A5B4FC",  
    "Tooth Location and Counting": "#3B82F6", 
}
color_map_Second_task_yellow_soft = {
    "Abnormality Diagnosis": "#F6BE4F",     # 柔和亮金黄（主色，温柔不刺眼）
    "Treatment Planning": "#FFDB6E",        # 温润的柠檬黄（略亮，用于重点）
    "Examination Question": "#FFE999",      # 适中暖黄，有视觉核心
    "Cervical vertebral maturation (CVM) stage Prediction": "#D9A441",  # 稍偏橙调，平衡过渡
    "Dental treatment video comprehension": "#E9B44C",  # 稍深但温暖
    "Tooth Location and Counting": "#FFD35C",           # 压轴暖黄，保持协调
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
                inner_colors.append(color_map_First_modality_red.get(key, "#aaaaaa"))
        
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
plt.savefig('dual_layer_donut_chart.png', bbox_inches='tight', pad_inches=0.01, dpi=300)