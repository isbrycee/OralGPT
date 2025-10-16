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
    {"Teeth": 36.16, "Patho": 41.14, "His": 36.18, "Jaw": 47.85, "SumRec": 57.89}, # gpt-4o
    {"Teeth": 29.41, "Patho": 31.82, "His": 33.33, "Jaw": 51.70, "SumRec": 42.22}, # Deepseek-vl-7b
    {"Teeth": 24.60, "Patho": 24.68, "His": 27.16, "Jaw": 26.79, "SumRec": 42.22}, # Qwen2.5-VL-72B
    {"Teeth": 30.09, "Patho": 32.92, "His": 30.54, "Jaw": 38.20, "SumRec": 60.42}, # LLaVA-NeXT-13B-hf
    {"Teeth": 38.15, "Patho": 36.36, "His": 43.83, "Jaw": 72.45, "SumRec": 71.11}, # Ovis2-34B
    {"Teeth": 26.56, "Patho": 22.08, "His": 22.22, "Jaw": 33.58, "SumRec": 28.89}, # InternVL3-38B
    # {"Teeth": 34.16, "Patho": 32.30, "His": 36.53, "Jaw": 71.91, "SumRec": 62.50}, # mPLUG-Owl3-7B
    {"Teeth": 26.56, "Patho": 21.43, "His": 24.65, "Jaw": 38.49, "SumRec": 22.22}, # MedDr
    {"Teeth": 23.17, "Patho": 23.38, "His": 17.28, "Jaw": 29.81, "SumRec": 35.56}, # Kimi-VL-A3B-Thinking
    {"Teeth": 37.17, "Patho": 30.43, "His": 38.32, "Jaw": 52.81, "SumRec": 45.83}, # Qwen2.5-VL-7B (SFT)
]

labels = ["Teeth", "Patho", "HisT", "Jaw", "SumRec"]
models = [
    "GPT-4-Turbo",
    "Deepseek-vl-7B",
    "Qwen2.5-VL-72B",
    "LLaVA-NeXT-13B-hf",
    "Ovis2-34B",
    "InternVL3-38B",
    "MedDr",
    "Kimi-VL-A3B-Thinking",
    "Qwen2.5-VL-7B (SFT)",
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
ax.set_ylim(0, 83)

# Add title
# plt.title("Radar Chart of Model Performance", size=16, fontweight='bold', pad=20)

# Add legend
ax.legend(loc="best",  fontsize=17,)

# Save the figure (for use in papers)
plt.tight_layout()
plt.savefig("radar_chart_closed_new.png", dpi=300, bbox_inches='tight')

# Show the plot
# plt.savefig("petr-leidatu_v6_eps.png", format="png",transparent=False)
