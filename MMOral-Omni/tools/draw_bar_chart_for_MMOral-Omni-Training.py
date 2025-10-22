import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Times New Roman'

def plot_horizontal_bar(data: dict, output_path: str = "modality_distribution.png"):
    """
    绘制学术级横向柱状图（数量最多的模态在最上端，按降序排列）
    
    参数:
        data (dict): 输入数据，key=图像模态，value=数量
        output_path (str): 输出PNG文件路径
    """
    # ---------------------- 1. 数据预处理：按数量降序排序（从多到少）
    # 按值降序排列（reverse=True），确保模态从多到少排列
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    sorted_modalities = [item[0] for item in sorted_items]  # 降序后的模态（如[A,B,C]，A最多）
    sorted_counts = [item[1] for item in sorted_items]      # 降序后的数量（如[10,5,3]）
    
    # ---------------------- 2. 初始化图表：高分辨率+学术字体
    plt.figure(figsize=(10, 6), dpi=300)  # 画布尺寸（宽10in，高6in），300DPI高分辨率
    # plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]  # 学术通用无衬线字体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
    
    sns.set(style="whitegrid", font="Times New Roman")
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(sorted_modalities)))

    # ---------------------- 3. 绘制横向柱状图：Viridis渐变颜色（学术友好）
    bars = plt.barh(
        y=sorted_modalities,  # Y轴使用降序后的模态（A在最下，C在最上）
        width=sorted_counts,  # X轴为数量
        # color=plt.cm.viridis(np.linspace(1, 0, len(sorted_modalities))),  # Viridis渐变（色盲友好）
        color=colors,
        edgecolor="none",    # 柱子白色边框，增强区分度
        linewidth=0.8         # 边框粗细
    )
    
    # ---------------------- 4. 添加数值标签（千分位格式，右端对齐）
    max_count = max(sorted_counts)
    for idx, (modality, count) in enumerate(zip(sorted_modalities, sorted_counts)):
        plt.text(
            x=count + max_count * 0.01,  # 标签位置：柱子右端+1%偏移（避免重叠）
            y=idx,                       # 对应模态的Y坐标
            s=f"{count:,}",              # 千分位格式（如15,000）
            va="center",                 # 垂直居中
            ha="left",                   # 水平左对齐
            fontsize=9,                  # 字体大小
            fontweight="bold",           # 粗体增强可读性
            color="#333333"              # 深灰文本颜色
        )
    
    # ---------------------- 5. 美化细节：学术图表标准配置
    # plt.xlabel("数量", fontsize=12, fontweight="bold")       # X轴标签（粗体）
    # plt.ylabel("图像模态", fontsize=12, fontweight="bold")   # Y轴标签（粗体）
    # plt.title("图像模态数量分布", fontsize=14, fontweight="bold", pad=15)  # 标题（pad避免重叠）
    plt.xlim(0, max_count * 1.1)                              # X轴留10%空白（避免标签贴边）
    plt.grid(axis="x", alpha=0.5, linestyle="--")             # X轴浅灰虚线网格（辅助数值定位）
    ax.grid(False, axis="y")
    plt.gca().invert_yaxis()                                  # 关键！反转Y轴，让最多的模态到顶部
    plt.yticks(rotation=12)

     # 去除图框四条边，仅保留底线和左线（学术风格）
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)
    # ax.spines["left"].set_color("#888888")
    # ax.spines["bottom"].set_color("#888888")
    # ax.spines["left"].set_linewidth(0.5)
    # ax.spines["bottom"].set_linewidth(0.5)

    # ---------------------- 6. 保存图表：高分辨率+去白边
    plt.tight_layout()  # 自动调整布局，避免标签被裁剪
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # 保存为PNG（300DPI，去白边）
    print(f"图表已保存至：{output_path}（分辨率300 DPI，适合学术论文）")


# ---------------------- 示例调用（替换为您的实际数据）
if __name__ == "__main__":
    # 替换为您的图像模态数据（key=模态名称，value=数量）
    modality_number_dict = {
        "Text-Only Instruction Data": 4295, # 6678 indicates len(pt) + len(sft)
        # "Figures from Textbooks": 6318,
        "Intraoral Image": 14994,
        "Periapical Radiograph": 16396,
        "Panoramic Radiograph": 13000,
        "Cephalometric Radiograph": 802,
        "Pathological Image": 7767,
        "Intraoral Video": 90,
        "Interleaved Image-Text Data": 327,
        "3D Model Scan": 136
    }
    
    # 调用函数绘图（输出路径可自定义）
    plot_horizontal_bar(modality_number_dict, output_path="MMoral-Omni-Training-SFT-Dist.png")