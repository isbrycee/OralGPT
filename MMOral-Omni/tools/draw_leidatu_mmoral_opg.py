import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import font_manager

# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


font_path = 'xkcd-script.ttf'
try:
    # 添加字体到字体管理器
    font_manager.fontManager.addfont(font_path)
    
    # 获取字体名称（自动解析）
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    
    # 设置全局字体
    plt.rcParams['font.family'] = font_name
    print(f"字体加载成功: {font_name}")

except Exception as e:
    print(f"字体加载失败: {str(e)}")
    print("请检查文件路径是否正确，或使用备用字体")
    plt.rcParams['font.family'] = 'sans-serif'  # 退回默认字体
# 设置字体为 Times New Roman（或其他您喜欢的字体）

rc('font', family='xkcd Script')

# plt.rcParams['font.sans-serif']=['DejaVu Sans']     #显示中文
# plt.rcParams['axes.unicode_minus']=False       #正常显示负号


import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Data
results = [
    {"Teeth": 31.46, "Patho": 23.79, "His": 39.51, "Jaw": 69.81, "SumRec": 34.29, "Report": 43.70, "average": 39.38},  # GPT-4V
    {"Teeth": 28.04, "Patho": 24.77, "His": 31.90, "Jaw": 47.81, "SumRec": 12.98, "Report": 16.70, "average": 27.84},  # Gemini-2.5-Flash
    {"Teeth": 2.10, "Patho": 4.47, "His": 7.06, "Jaw": 11.62, "SumRec": 7.98, "Report": 5.50, "average": 5.29},      # Qwen-Max-VL
    {"Teeth": 16.48, "Patho": 7.50, "His": 13.44, "Jaw": 34.56, "SumRec": 9.52, "Report": 9.60, "average": 15.95},    # Deepseek-VL-7b-chat
    {"Teeth": 20.94, "Patho": 9.70, "His": 18.77, "Jaw": 26.62, "SumRec": 12.74, "Report": 21.30, "average": 19.74},  # GLM-4V-9B
    {"Teeth": 13.90, "Patho": 15.83, "His": 15.40, "Jaw": 27.12, "SumRec": 7.38, "Report": 11.50, "average": 15.38},  # Qwen2.5-VL-72B
    # {"Teeth": 0.91, "Patho": 1.52, "His": 0.00, "Jaw": 0.00, "SumRec": 0.00, "Report": 24.50, "average": 4.76},      # LLaVA-Med
    {"Teeth": 30.64, "Patho": 25.83, "His": 27.98, "Jaw": 51.12, "SumRec": 17.02, "Report": 8.00, "average": 27.80},  # HealthGPT-XL32
    {"Teeth": 22.42, "Patho": 13.71, "His": 24.42, "Jaw": 43.88, "SumRec": 13.57, "Report": 25.80, "average": 24.70},  # MedVLM-R1
    # {"Teeth": 22.99, "Patho": 32.58, "His": 29.57, "Jaw": 52.44, "SumRec": 20.95, "Report": 8.70, "average": 26.20},   # MedDr
    {"Teeth": 37.76, "Patho": 30.91, "His": 40.31, "Jaw": 54.69, "SumRec": 43.93, "Report": 29.90, "average": 38.86},  # OralGPT-Omni (Ours)
]

labels = ["Teeth", "Patho", "HisT", "Jaw", "SumRec", "Report", "Overall"]

models = [
    "GPT-4V",                # 1
    "Gemini-2.5-Flash",      # 2
    "Qwen-Max-VL",           # 3
    "Deepseek-VL-7b-chat",   # 4
    "GLM-4V-9B",             # 5
    "Qwen2.5-VL-72B",        # 6
    # "LLaVA-Med",             # 7
    "HealthGPT-XL32",        # 8
    "MedVLM-R1",             # 9
    # "MedDr",                 # 10
    "OralGPT-Omni (Ours)",   # 11
]



# Prepare data for radar chart
num_vars = len(labels)
# angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
# angles += angles[:1]  # Complete the loop

# Rotate the radar chart by changing the starting angle
start_angle = 39  # Adjust this to rotate the chart (0-360 degrees)
angles = [((n / float(num_vars)) * 2 * pi + np.radians(start_angle)) % (2 * pi) for n in range(num_vars)]
angles += angles[:1]  # Complete the loop


# Initialize the radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
cmap = plt.cm.get_cmap("Set1", len(models))
colors = cmap(np.arange(len(models)))[::-1]

# Plot each model
for idx, result in enumerate(results):
    values = list(result.values())
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, label=models[idx], linewidth=2, linestyle='solid', color=colors[idx])
    ax.fill(angles, values, alpha=0.25, color=colors[idx])

# Add labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=25)

# Set the range for the radial axis
ax.set_rscale("linear")
ax.set_rlabel_position(120)
ax.set_yticks([20, 40, 60, 80])  # Customize tick positions
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=20, color="gray")
ax.set_ylim(0, 70)

# Add title
# plt.title("Radar Chart of Model Performance", size=16, fontweight='bold', pad=20)

# Add legend
ax.legend(loc="lower right",  fontsize=17,)

# Save the figure (for use in papers)
plt.tight_layout()
plt.savefig("radar_chart_MMOral-OPG.png", dpi=300, bbox_inches='tight')

# Show the plot
# plt.savefig("petr-leidatu_v6_eps.png", format="png",transparent=False)

