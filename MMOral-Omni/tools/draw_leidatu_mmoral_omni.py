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


font_path = './xkcd-script.ttf'
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
    {"II": 38.33, "Loc": 44.60, "Dx-I": 45.24, "Dx-R": 25.16, "PA": 31.43, "CE": 41.27, "PI": 40.52, "TP": 80.67, "IV": 56.00, "Overall": 36.42},  # GPT-5
    {"II": 42.23, "Loc": 45.00, "Dx-I": 53.49, "Dx-R": 28.19, "PA": 37.48, "CE": 28.60, "PI": 37.52, "TP": 75.33, "IV": 50.00, "Overall": 38.70},  # o3
    {"II": 33.93, "Loc": 31.00, "Dx-I": 51.10, "Dx-R": 19.70, "PA": 37.01, "CE": 36.30, "PI": 35.09, "TP": 73.33, "IV": 46.00, "Overall": 35.72},  # Gemini-2.5-flash
    {"II": 22.41, "Loc": 11.80, "Dx-I": 35.71, "Dx-R": 20.27, "PA": 30.89, "CE": 22.60, "PI": 34.73, "TP": 72.67, "IV": 23.00, "Overall": 28.47},  # Grok-4
    {"II": 27.06, "Loc": 21.70, "Dx-I": 45.10, "Dx-R": 14.37, "PA": 38.37, "CE": 31.00, "PI": 27.60, "TP": 70.00, "IV": 33.00, "Overall": 31.05},  # GLM-4.5v
    {"II": 28.81, "Loc": 36.30, "Dx-I": 38.85, "Dx-R": 11.28, "PA": 38.03, "CE": 33.33, "PI": 34.52, "TP": 79.33, "IV": 29.00, "Overall": 30.32},  # Claude-Sonnet-4-5-20250929
    {"II": 23.74, "Loc": 18.70, "Dx-I": 40.00, "Dx-R": 12.53, "PA": 33.62, "CE": 26.70, "PI": 28.75, "TP": 56.00, "IV": 23.00, "Overall": 27.83},  # Qwen3-VL-235B-A22B
    {"II": 17.32, "Loc": 10.10, "Dx-I": 30.85, "Dx-R": 11.02, "PA": 32.95, "CE": 14.80, "PI": 28.67, "TP": 64.67, "IV": 14.00, "Overall": 23.39},  # InternVL3.5-8B
    {"II": 54.46, "Loc": 66.80, "Dx-I": 56.60, "Dx-R": 39.99, "PA": 48.11, "CE": 65.90, "PI": 56.01, "TP": 47.33, "IV": 66.00, "Overall": 51.84},  # OralGPT-Omni (Ours)
]

for item in results:
    item.pop("TP")  # Remove "TP" from each dictionary

labels = ["II", "Loc", "Dx-I", "Dx-R", "PA", "CE", "PI", "IV", "Overall"] # "TP", 
models = [
    "GPT-5",
    "o3",
    "Gemini-2.5-flash",
    "Grok-4",
    "GLM-4.5v",
    "Claude-Sonnet-4-5",
    "Qwen3-VL-235B-A22B",
    "InternVL3.5-8B",
    "OralGPT-Omni (Ours)",
]

# Prepare data for radar chart
num_vars = len(labels)
# angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
# angles += angles[:1]  # Complete the loop

# Rotate the radar chart by changing the starting angle
start_angle = 130  # Adjust this to rotate the chart (0-360 degrees)
angles = [((n / float(num_vars)) * 2 * pi + np.radians(start_angle)) % (2 * pi) for n in range(num_vars)]
angles += angles[:1]  # Complete the loop


# Initialize the radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
colors = plt.cm.get_cmap("Set1", len(models))
colors = colors(np.arange(len(models)))[::-1]
# Plot each model
for idx, result in enumerate(results):
    values = list(result.values())
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, label=models[idx], linewidth=2, linestyle='solid', color=colors[idx])
    ax.fill(angles, values, alpha=0.25, color=colors[idx])

# Add labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=35)

# Set the range for the radial axis
ax.set_rscale("linear")
ax.set_rlabel_position(120)
ax.set_yticks([20, 40, 60, 80])  # Customize tick positions
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=30, color="gray")
ax.set_ylim(0, 70)

# Add title
# plt.title("Radar Chart of Model Performance", size=16, fontweight='bold', pad=20)

# Add legend
ax.legend(loc="best",  fontsize=23,)

# Save the figure (for use in papers)
plt.tight_layout()
plt.savefig("radar_chart_MMOral-Omni.png", dpi=100, bbox_inches='tight')

# Show the plot
# plt.savefig("petr-leidatu_v6_eps.png", format="png",transparent=False)
