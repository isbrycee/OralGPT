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
modality_number_dict = {
    "Plain Text": 6678,
    "Figures from Textbooks": 6318,
    "Intraoral Image (Location and Counting)": 756,
    "Intraoral Image (Image-level Analysis)": 4803,
    "Intraoral Image (Region-level Analysis)": 9435,
    "Periapical Radiograph": 16396,
    "Panoramic Radiograph": 20563,
    "Cephalometric Radiograph": 802,
    "Histopathological Image": 7768,
    "Intraoral Video": 90,
    "Interleaved Image-Text data": 327,
    "3D Model Scan": 136
}

# -----------------------------
# 配色方案
# -----------------------------
# 优化的配色方案 - 融入蓝色主题
# -----------------------------
color_map = {
    "Plain Text": "#687EFF",        # 您的深蓝色
    "Figures from Textbooks": "#80B3FF",    # 您的中蓝色
    "Intraoral Image": "#98E4FF",   # 您的浅蓝色 - 三个子任务用相同颜色
    "Periapical Radiograph": "#FF6B6B",     # 暖红色，与蓝色形成对比
    "Panoramic Radiograph": "#FFD166",      # 暖黄色
    "Cephalometric Radiograph": "#06D6A0",  # 青绿色
    "Histopathological Image": "#B6FFFA",   # 您的极浅蓝色
    "Intraoral Video": "#C8B6FF",           # 淡紫色
    "Interleaved Image-Text data": "#FFA69E", # 珊瑚粉色
    "3D Model Scan": "#A0C4FF",             # 另一种蓝色调
}


# -----------------------------
# 数据准备
# -----------------------------
labels = []
values = []
colors = []

for key, val in modality_number_dict.items():
    labels.append(key)
    values.append(val)
    if key.startswith("Intraoral Image"):
        colors.append(color_map["Intraoral Image"])
    else:
        main_label = key.split(" (")[0]
        colors.append(color_map.get(main_label, "#aaaaaa"))

# -----------------------------
# 绘制饼图
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

wedges, _ = ax.pie(
    values,
    colors=colors,
    startangle=90,
    wedgeprops=dict(width=0.5, edgecolor='white')  # 中心空白形成环形
)

# 计算比例并在 >1% 的部分添加文字
total = sum(values)
for i, wedge in enumerate(wedges):
    ratio = 100 * values[i] / total
    if ratio > 0.99:
        # 计算角度和中心点
        theta = (wedge.theta2 + wedge.theta1) / 2.0
        x = 0.7 * np.cos(np.deg2rad(theta))
        y = 0.7 * np.sin(np.deg2rad(theta))
        ax.text(x, y, f"{ratio:.1f}%", ha='center', va='center',
                fontsize=18, fontweight='bold', fontfamily='Times New Roman', color='black')

ax.axis('equal')
# plt.title("Modality Distribution (Ring Chart)", fontsize=14, fontfamily='Times New Roman')

# -----------------------------
# 打印每个模态数量占比
# -----------------------------
print("各模态数量占比：")
for label, value in zip(labels, values):
    ratio = value / total * 100
    print(f"{label:45s}: {value:6d} ({ratio:6.2f}%)")

plt.tight_layout()
plt.savefig('output.png')  # 保存图像到当前目录下的 output.png 文件

# plt.show()