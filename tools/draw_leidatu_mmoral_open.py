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


font_path = '/Users/jinghao/Documents/hku/Project/X-ray_VLM/xkcd-script.ttf'
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
    {"Teeth": 41.99, "Patho": 27.20, "His": 41.96, "Jaw": 66.00, "SumRec": 41.79, "Report": 60.40}, # gpt-4o
    {"Teeth": 16.48, "Patho": 7.50, "His": 13.44, "Jaw": 34.56, "SumRec": 9.52, "Report": 9.60}, # Deepseek-vl-7b
    {"Teeth": 13.90, "Patho": 15.83, "His": 15.40, "Jaw": 27.12, "SumRec": 7.38, "Report": 11.50}, # Qwen2.5-VL-72B
    {"Teeth": 14.48, "Patho": 10.28, "His": 9.23, "Jaw": 22.41, "SumRec": 14.30, "Report": 21.30}, # LLaVA-NeXT-13B-hf
    {"Teeth": 35.21, "Patho": 22.12, "His": 37.79, "Jaw": 55.31, "SumRec": 16.43, "Report": 32.20}, # Ovis2-34B
    {"Teeth": 34.77, "Patho": 19.17, "His": 30.18, "Jaw": 47.69, "SumRec": 17.74, "Report": 40.10}, # InternVL3-38B
    # {"Teeth": 12.50, "Patho": 8.44, "His": 8.52, "Jaw": 30.59, "SumRec": 3.26, "Report": 13.90,}, # mPLUG-Owl3-7B
    {"Teeth": 22.99, "Patho": 32.58, "His": 29.57, "Jaw": 52.44, "SumRec": 20.95, "Report": 8.7,}, # MedDr
    {"Teeth": 50.39, "Patho": 37.73, "His": 50.18, "Jaw": 58.25, "SumRec": 45.71, "Report": 61.50,}, # Kimi-VL-A3B-Thinking
    {"Teeth": 55.45, "Patho": 33.40, "His": 45.74, "Jaw": 74.47, "SumRec": 45.17, "Report": 50.5,}, # Qwen2.5-VL-7B SFT
]

labels = ["Teeth", "Patho", "HisT", "Jaw", "SumRec", "Report"]
models = [
    "GPT-4o",
    "Deepseek-vl-7B",
    "Qwen2.5-VL-72B",
    "LLaVA-NeXT-13B-hf",
    "Ovis2-34B",
    "InternVL3-38B",
    "MedDr",
    "Kimi-VL-A3B-Thinking",
    "Qwen2.5-VL-7B (SFT)"
]

# Prepare data for radar chart
num_vars = len(labels)
# angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
# angles += angles[:1]  # Complete the loop

# Rotate the radar chart by changing the starting angle
start_angle = 17  # Adjust this to rotate the chart (0-360 degrees)
angles = [((n / float(num_vars)) * 2 * pi + np.radians(start_angle)) % (2 * pi) for n in range(num_vars)]
angles += angles[:1]  # Complete the loop


# Initialize the radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
colors = plt.cm.get_cmap("Set1", len(models))

# Plot each model
for idx, result in enumerate(results):
    values = list(result.values())
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, label=models[idx], linewidth=2, linestyle='solid', color=colors(idx))
    ax.fill(angles, values, alpha=0.25, color=colors(idx))

# Add labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=25)

# Set the range for the radial axis
ax.set_rscale("linear")
ax.set_rlabel_position(120)
ax.set_yticks([20, 40, 60, 80])  # Customize tick positions
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=20, color="gray")
ax.set_ylim(0, 75)

# Add title
# plt.title("Radar Chart of Model Performance", size=16, fontweight='bold', pad=20)

# Add legend
ax.legend(loc="best",  fontsize=17,)

# Save the figure (for use in papers)
plt.tight_layout()
plt.savefig("radar_chart_open_new.png", dpi=300, bbox_inches='tight')

# Show the plot
# plt.savefig("petr-leidatu_v6_eps.png", format="png",transparent=False)
