from matplotlib import rc
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# 设置字体为 Times New Roman（或其他您喜欢的字体）
rc('font', family='Times New Roman')

# plt.rcParams['font.sans-serif']=['DejaVu Sans']     #显示中文
# plt.rcParams['axes.unicode_minus']=False       #正常显示负号


# Data
results = [
    {"Teeth": 37.88, "Patho": 39.13, "His": 48.50, "Jaw": 51.69, "SumRec": 58.33}, # gpt-4o
    {"Teeth": 22.65, "Patho": 17.39, "His": 28.74, "Jaw": 59.93, "SumRec": 52.08}, # Deepseek-vl-7b
    {"Teeth": 26.55, "Patho": 27.95, "His": 26.35, "Jaw": 22.47, "SumRec": 47.92}, # Qwen2.5-VL-72B
    {"Teeth": 30.09, "Patho": 32.92, "His": 30.54, "Jaw": 38.20, "SumRec": 60.42}, # LLaVA-NeXT-13B-hf
    {"Teeth": 28.67, "Patho": 21.12, "His": 25.75, "Jaw": 39.33, "SumRec": 31.25}, # InternVL3-38B
    {"Teeth": 45.84, "Patho": 51.55, "His": 53.89, "Jaw": 79.40, "SumRec": 70.17}, # Ovis2-34B
    {"Teeth": 34.16, "Patho": 32.30, "His": 36.53, "Jaw": 71.91, "SumRec": 62.50}, # mPLUG-Owl3-7B
]

labels = ["Teeth", "Patho", "His", "Jaw", "SumRec"]
models = [
    "GPT-4o",
    "Deepseek-vl-7b",
    "Qwen2.5-VL-72B",
    "LLaVA-NeXT-13B-hf",
    "InternVL3-38B",
    "Ovis2-34B",
    "mPLUG-Owl3-7B",
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
ax.set_xticklabels(labels, fontsize=12)

# Set the range for the radial axis
ax.set_rscale("linear")
ax.set_rlabel_position(30)
ax.set_yticks([20, 40, 60, 80])  # Customize tick positions
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=14, color="gray")
ax.set_ylim(0, 83)

# Add title
# plt.title("Radar Chart of Model Performance", size=16, fontweight='bold', pad=20)

# Add legend
ax.legend(loc="best",  fontsize=10,)

# Save the figure (for use in papers)
plt.tight_layout()
plt.savefig("radar_chart.png", dpi=300, bbox_inches='tight')

# Show the plot
# plt.savefig("petr-leidatu_v6_eps.png", format="png",transparent=False)
