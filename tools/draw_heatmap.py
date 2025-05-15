import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "GPT-4o": "39.65 40.99 46.71 55.81 56.25 45.40 31.48 26.05 37.56 57.42 30.37 42.50 37.50 41.45",
    "Gemini-2.0-Flash": "37.17 35.40 44.31 46.82 58.33 41.20 37.27 26.05 40.36 52.40 35.05 49.00 40.67 40.94",
    "Ovis2-34B": "45.84 51.55 53.89 79.40 79.17 56.80 32.48 24.33 31.60 50.88 21.05 31.70 33.02 44.91",
    "Kimi-VL-A3B-Instruct": "44.60 42.24 38.32 70.79 66.67 49.8 28.20 17.94 32.49 53.53 19.95 25.60 30.00 39.90",
    "Kimi-VL-A3B-Thinking": "25.84 27.33 25.75 29.96 27.08 26.80 52.53 37.66 53.79 68.59 50.93 61.5 54.55 40.68",
    "mPLUG-Owl3-7B": "34.16 32.30 36.53 71.91 62.50 42.80 12.50 8.44 8.52 30.59 3.26 13.90 13.67 28.24",
    "Ovis2-8B": "40.18 47.83 44.91 75.66 70.83 50.80 28.70 26.17 26.63 53.06 20.81 30.72 30.70 40.75",
    "MedDr": "36.46 36.02 41.92 73.03 64.58 46.00 27.50 28.14 30.20 49.17 26.17 7.50 26.17 36.09",
    "MedVLM-R1": "28.67 31.68 37.72 65.17 47.92 38.60 22.58 12.28 21.57 40.61 21.96 24.50 24.58 31.59",
    "Deepseek-vl-7b": "22.65 17.39 28.74 59.93 52.08 31.20 12.75 8.16 8.40 30.00 13.14 9.10 13.42 22.31",
    "Cambrian-34B": "34.87 34.16 44.31 70.04 60.42 44.40 33.10 21.42 31.83 48.24 13.60 16.00 29.63 37.02",
    "Phi-4-multimodal-instruct": "36.28 36.65 49.10 60.30 54.17 43.60 25.52 20.14 27.69 43.29 13.84 12.80 24.57 34.09",
    "GLM-4V-9B": "29.03 35.40 41.32 62.55 64.58 40.20 17.85 8.01 17.46 24.12 15.93 19.40 17.50 28.85",
    "HuatuoGPT-V-34B": "28.85 14.91 29.94 26.59 22.92 25.60 32.62 18.65 28.05 53.12 18.60 15.40 29.48 27.54",
    "CogVLM2-19B": "33.63 31.68 34.13 38.95 60.42 35.20 26.11 17.09 26.86 49.24 18.14 24.50 27.63 31.42",
    "MiniCPM-O2.6": "30.27 23.60 24.55 36.33 14.58 30.27 29.20 17.38 24.38 49.76 15.93 27.90 28.42 28.21",
    "Phi-3-Vision-128K-Instruct": "30.62 32.92 40.72 44.57 64.58 37.40 20.18 17.80 16.57 46.35 20.35 8.60 20.93 29.17",

    "InternVL3-38B": "28.67 21.12 25.75 39.33 31.25 28.40 33.69 22.41 29.70 46.11 20.23 42.90 34.15 31.28",
    "LLaVA-OneVision": "14.51 18.01 35.33 42.70 31.25 24.40 22.68 13.48 17.75 38.35 18.72 11.20 20.93 22.67",
    "Yi-VL-34B": "36.81 36.64 43.11 41.20 70.83 40.20 24.97 23.40 20.59 39.35 15.23 9.90 22.98 31.59",
    "Gemma3-12B": "24.78 19.88 31.74 34.08 29.17 26.60 25.21 20.00 20.65 26.88 22.33 33.20 25.32 25.96",
    "Chameleon-7b": "32.57 44.10 37.13 29.59 52.08 35.80 6.02 6.10 9.35 9.71 5.35 8.40 7.27 21.54",
    "Qwen2.5-VL-72B": "26.55 27.95 26.35 22.47 47.92 26.80 13.05 18.44 11.66 26.88 7.44 11.50 14.77 20.79",
    
    
    "LLaVA-Med": "25.49 26.71 21.56 13.86 45.83 23.20 23.23 18.75 11.36 32.82 26.28 5.30 19.60 21.40",
    "LLaVA-NeXT-13B-hf": "30.09 32.92 30.54 38.20 60.42 33.80 14.48 10.28 9.23 22.41 14.30 21.30 15.43 24.62",
    "Qwen2.5-VL-7B": "24.96 21.12 27.54 37.08 35.42 27.00 17.01 16.10 11.18 29.41 9.07 8.20 15.92 21.46",
    "LLaVA-v1.5-7B": "20.53 17.39 28.14 26.22 47.92 23.00 12.57 13.19 10.18 18.88 17.09 11.50 13.22 18.11",
    "Molmo-72B-0924": "28.85 14.91 29.94 26.59 22.92 25.60 9.25 6.31 3.49 12.65 5.00 9.20 8.23 16.92",
    "Kosmos-2": "15.75 18.01 28.14 10.11 25.00 17.40 13.58 10.71 11.18 19.76 8.49 3.40 11.87 14.64",

    "XComposer2-vl-7b": "5.49 26.71 21.56 13.86 45.83 23.20 6.52 11.01 15.00 7.67 2.10 8.53 8.99 16.10",
    

}

# Preprocess data
processed_data = {
    model: list(map(float, values.split(" ")))
    for model, values in data.items()
}
processed_data = {
    key: values[:4] + values[5:-2]  # Keep the first 4 elements, skip the 5th, and exclude the last two
    for key, values in processed_data.items()
}
processed_data = {
        model: [round(x/100, 4) for x in values]  # 除以100并保留4位小数
        for model, values in processed_data.items()
    }

# Convert to DataFrame
df = pd.DataFrame(processed_data).T

# Fill missing values with NaN (if lengths are uneven)
df = df.apply(lambda row: pd.Series(row.dropna().tolist()), axis=1)
custom_xticks = ['Teeth', 'Patho', 'HisT', 'Jaw', 'Summ', 'Teeth', 'Patho', 'HisT', 'Jaw', 'Summ', 'Report']
# Create a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(
    df,
    annot=False,
    cmap="coolwarm",
    cbar=True,
    xticklabels=custom_xticks,
    yticklabels=True,
    linewidths=1,
    vmin=0,
    vmax=1,
)

# 在第五列后面画一条垂直分隔线
col_index = 5  # 第五列后分隔线（注意：索引从 1 开始）
plt.axvline(x=col_index, color="white", linestyle="-", linewidth=6)

# Labels and adjustments
plt.xticks(rotation=45, ha="center", fontsize=10)
plt.yticks(fontsize=10)
plt.text(0.11, 1.01, 'Closed-Ended', transform=plt.gca().transAxes, fontweight='bold', fontsize=12)
plt.text(0.63, 1.01, 'Open-Ended', transform=plt.gca().transAxes, fontweight='bold', fontsize=12)
# plt.title("Comparison Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("heatmap.svg", dpi=300, bbox_inches='tight', format='svg')


# # Show the plot
# plt.show()