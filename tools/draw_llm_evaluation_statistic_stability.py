import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pingouin import intraclass_corr
from scipy.stats import f_oneway

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

gpt_4o_5times_results = {
    "evaluation_0": {
        "Teeth": {"tot": 589, "acc": 31.48},
        "Patho": {"tot": 167, "acc": 26.05},
        "HisT": {"tot": 197, "acc": 37.56},
        "Jaw": {"tot": 229, "acc": 57.42},
        "SumRec": {"tot": 107, "acc": 30.37},
        "Report": {"tot": 300, "acc": 42.50},
        "Overall": {"tot": 600, "acc": 37.500000}
    },
    "evaluation_1": {
        "Teeth": {"tot": 589, "acc": 32.512733},
        "Patho": {"tot": 167, "acc": 25.449102},
        "HisT": {"tot": 197, "acc": 36.294416},
        "Jaw": {"tot": 229, "acc": 53.930131},
        "SumRec": {"tot": 107, "acc": 34.112150},
        "Report": {"tot": 300, "acc": 43.000000},
        "Overall": {"tot": 600, "acc": 37.500000}
    },
    "evaluation_2": {
        "Teeth": {"tot": 589, "acc": 31.578947},
        "Patho": {"tot": 167, "acc": 26.946108},
        "HisT": {"tot": 197, "acc": 34.010152},
        "Jaw": {"tot": 229, "acc": 53.493450},
        "SumRec": {"tot": 107, "acc": 34.112150},
        "Report": {"tot": 300, "acc": 43.500000},
        "Overall": {"tot": 600, "acc": 37.166667}
    },
    "evaluation_3": {
        "Teeth": {"tot": 589, "acc": 32.173175},
        "Patho": {"tot": 167, "acc": 26.347305},
        "HisT": {"tot": 197, "acc": 35.279188},
        "Jaw": {"tot": 229, "acc": 55.458515},
        "SumRec": {"tot": 107, "acc": 35.981308},
        "Report": {"tot": 300, "acc": 42.000000},
        "Overall": {"tot": 600, "acc": 37.583333}
    },
    "evaluation_4": {
        "Teeth": {"tot": 589, "acc": 32.003396},
        "Patho": {"tot": 167, "acc": 24.550898},
        "HisT": {"tot": 197, "acc": 35.532995},
        "Jaw": {"tot": 229, "acc": 57.860262},
        "SumRec": {"tot": 107, "acc": 35.046729},
        "Report": {"tot": 300, "acc": 45.000000},
        "Overall": {"tot": 600, "acc": 38.083333}
    }
}


healthgpt_xl32_5times_results = {
    "evaluation_0": {
        "teeth": {"tot": 589, "acc": 29.80},
        "patho": {"tot": 167, "acc": 22.16},
        "his": {"tot": 197, "acc": 24.11},
        "jaw": {"tot": 229, "acc": 47.82},
        "summ": {"tot": 107, "acc": 24.77},
        "report": {"tot": 300, "acc": 10.00},
        "Overall": {"tot": 600, "acc": 27.17}
    },
    "evaluation_1": {
        "teeth": {"tot": 589, "acc": 31.748727},
        "patho": {"tot": 167, "acc": 23.353293},
        "his": {"tot": 197, "acc": 22.842640},
        "jaw": {"tot": 229, "acc": 47.161572},
        "summ": {"tot": 107, "acc": 28.971963},
        "report": {"tot": 300, "acc": 7.000000},
        "Overall": {"tot": 600, "acc": 27.583333}
    },
    "evaluation_2": {
        "teeth": {"tot": 589, "acc": 31.069610},
        "patho": {"tot": 167, "acc": 21.556886},
        "his": {"tot": 197, "acc": 25.380711},
        "jaw": {"tot": 229, "acc": 47.161572},
        "summ": {"tot": 107, "acc": 24.299065},
        "report": {"tot": 300, "acc": 7.500000},
        "Overall": {"tot": 600, "acc": 27.250000}
    },
    "evaluation_3": {
        "teeth": {"tot": 589, "acc": 31.154499},
        "patho": {"tot": 167, "acc": 22.754491},
        "his": {"tot": 197, "acc": 21.827411},
        "jaw": {"tot": 229, "acc": 46.943231},
        "summ": {"tot": 107, "acc": 27.102804},
        "report": {"tot": 300, "acc": 8.000000},
        "Overall": {"tot": 600, "acc": 27.166667}
    },
    "evaluation_4": {
        "teeth": {"tot": 589, "acc": 31.324278},
        "patho": {"tot": 167, "acc": 21.856287},
        "his": {"tot": 197, "acc": 21.827411},
        "jaw": {"tot": 229, "acc": 45.633188},
        "summ": {"tot": 107, "acc": 27.570093},
        "report": {"tot": 300, "acc": 9.500000},
        "Overall": {"tot": 600, "acc": 27.250000}
    }
}

qwen25_vl_7b_5times_results = {
    "evaluation_0": {
        "teeth": {"tot": 455, "acc": 17.01},
        "patho": {"tot": 141, "acc": 16.10},
        "his": {"tot": 169, "acc": 11.18},
        "jaw": {"tot": 170, "acc": 29.41},
        "summ": {"tot": 86, "acc": 9.07},
        "report": {"tot": 200, "acc": 8.20},
        "Overall": {"tot": 600, "acc": 15.92}
    },
    "evaluation_1": {
        "teeth": {"tot": 455, "acc": 17.4505},
        "patho": {"tot": 141, "acc": 16.0284},
        "his": {"tot": 169, "acc": 11.3609},
        "jaw": {"tot": 170, "acc": 27.9412},
        "summ": {"tot": 86, "acc": 9.65116},
        "report": {"tot": 200, "acc": 7.9},
        "Overall": {"tot": 600, "acc": 15.8833}
    },
    "evaluation_2": {
        "teeth": {"tot": 455, "acc": 17.2308},
        "patho": {"tot": 141, "acc": 16.383},
        "his": {"tot": 169, "acc": 11.1243},
        "jaw": {"tot": 170, "acc": 28.4118},
        "summ": {"tot": 86, "acc": 10.3488},
        "report": {"tot": 200, "acc": 8.2},
        "Overall": {"tot": 600, "acc": 15.9667}
    },
    "evaluation_3": {
        "teeth": {"tot": 455, "acc": 17.0769},
        "patho": {"tot": 141, "acc": 16.5957},
        "his": {"tot": 169, "acc": 11.716},
        "jaw": {"tot": 170, "acc": 28.1176},
        "summ": {"tot": 86, "acc": 10.2326},
        "report": {"tot": 200, "acc": 8.2},
        "Overall": {"tot": 600, "acc": 15.9667}
    },
    "evaluation_4": {
        "teeth": {"tot": 455, "acc": 17.0989},
        "patho": {"tot": 141, "acc": 16.383},
        "his": {"tot": 169, "acc": 11.716},
        "jaw": {"tot": 170, "acc": 27.8235},
        "summ": {"tot": 86, "acc": 9.76744},
        "report": {"tot": 200, "acc": 7.3},
        "Overall": {"tot": 600, "acc": 15.7333}
    }
}

ovis2_34b_5times_results = {
    "evaluation_0": {
        "teeth": {"tot": 455, "acc": 32.48},
        "patho": {"tot": 141, "acc": 24.33},
        "his": {"tot": 169, "acc": 31.60},
        "jaw": {"tot": 170, "acc": 50.88},
        "summ": {"tot": 86, "acc": 21.05},
        "report": {"tot": 200, "acc": 31.70},
        "Overall": {"tot": 600, "acc": 33.02}
    },
    "evaluation_1": {
        "teeth": {"tot": 455, "acc": 31.8681},
        "patho": {"tot": 141, "acc": 25.1064},
        "his": {"tot": 169, "acc": 30.4142},
        "jaw": {"tot": 170, "acc": 49.8235},
        "summ": {"tot": 86, "acc": 20.5814},
        "report": {"tot": 200, "acc": 30.9},
        "Overall": {"tot": 600, "acc": 32.4333}
    },
    "evaluation_2": {
        "teeth": {"tot": 455, "acc": 31.6264},
        "patho": {"tot": 141, "acc": 24.6099},
        "his": {"tot": 169, "acc": 29.5266},
        "jaw": {"tot": 170, "acc": 48.1765},
        "summ": {"tot": 86, "acc": 19.6512},
        "report": {"tot": 200, "acc": 31.9},
        "Overall": {"tot": 600, "acc": 32.0167}
    },
    "evaluation_3": {
        "teeth": {"tot": 455, "acc": 33.0989},
        "patho": {"tot": 141, "acc": 24.6099},
        "his": {"tot": 169, "acc": 30.2367},
        "jaw": {"tot": 170, "acc": 49.6471},
        "summ": {"tot": 86, "acc": 21.3953},
        "report": {"tot": 200, "acc": 31.5},
        "Overall": {"tot": 600, "acc": 32.9167}
    },
    "evaluation_4": {
        "teeth": {"tot": 455, "acc": 32.7912},
        "patho": {"tot": 141, "acc": 26.6667},
        "his": {"tot": 169, "acc": 31.4201},
        "jaw": {"tot": 170, "acc": 48.6471},
        "summ": {"tot": 86, "acc": 20.1163},
        "report": {"tot": 200, "acc": 31.7},
        "Overall": {"tot": 600, "acc": 32.9667}
    }
}

# Step 1: Extract accuracy values into a DataFrame
categories = list(gpt_4o_5times_results["evaluation_0"].keys())
runs = list(gpt_4o_5times_results.keys())

data = {category: [] for category in categories}
for run in runs:
    for category in categories:
        data[category].append(gpt_4o_5times_results[run][category]["acc"])

df = pd.DataFrame(data, index=runs)

# Step 2: Compute Mean, Standard Deviation, CV, and Range
results = {}
for category in categories:
    mean = df[category].mean()
    std = df[category].std()
    cv = (std / mean) * 100
    value_range = df[category].max() - df[category].min()

    results[category] = {
        "Mean": mean,
        "StdDev": std,
        "CV (%)": cv,
        "Range": value_range,
    }

stability_summary = pd.DataFrame(results).T
print("Stability Summary:")
print(stability_summary)

plt.figure(figsize=(8, 6))
# sns.heatmap(stability_summary, annot=True, cmap='viridis', fmt=".2f")
# plt.title('Stability Metrics Heatmap')
# plt.savefig("llm_eval_statistic.png")

stability_summary['Mean'].plot(kind='bar', yerr=stability_summary['StdDev'], capsize=5, color='#2DAA9E', edgecolor='black')
plt.ylabel('Mean Score ± StdDev (%)')
# plt.title('Mean Accuracy with Standard Deviation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("llm_eval_statistic_1.png")
