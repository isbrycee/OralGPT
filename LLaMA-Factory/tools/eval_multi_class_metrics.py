import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    classification_report,
    average_precision_score
)


gt_json = "/home/jinghao/projects/positioning-error/OPG_Positioning_Error_test.json"
pred_json = "/home/jinghao/projects/positioning-error/test_infer_results.json"


def compute_metrics(y_true, y_pred):
    num_classes = y_true.shape[1]
    num_samples = y_true.shape[0]

    # 1. 每个类别统计
    M_c = np.sum((y_true == 1) & (y_pred == 1), axis=0)  # 正确预测为正
    M_p = np.sum(y_pred == 1, axis=0)                   # 预测为正
    M_g = np.sum(y_true == 1, axis=0)                   # 真实为正

    # 2. Overall metrics
    OP = np.sum(M_c) / np.sum(M_p) if np.sum(M_p) > 0 else 0
    OR = np.sum(M_c) / np.sum(M_g) if np.sum(M_g) > 0 else 0
    OF1 = (2 * OP * OR) / (OP + OR) if (OP + OR) > 0 else 0

    # 3. Per-category metrics
    CP = np.mean(M_c / np.where(M_p > 0, M_p, 1))
    CR = np.mean(M_c / np.where(M_g > 0, M_g, 1))
    CF1 = (2 * CP * CR) / (CP + CR) if (CP + CR) > 0 else 0

    # 4. mAP (mean average precision)
    # average_precision_score 需要每个类别是二分类
    # 当y_pred为0/1时可以直接用
    try:
        mAP = average_precision_score(y_true, y_pred, average='macro')
    except Exception as e:
        mAP = np.nan

    return {
        'OP': OP,
        'OR': OR,
        'OF1': OF1,
        'CP': CP,
        'CR': CR,
        'CF1': CF1,
        'mAP': mAP
    }

# 1. 定义类别体系
CLASSES = [
    "Chin Tipped High",
    "Chin Tipped Low",
    "Head Tilted to the Left",
    "Head Tilted to the Right",
    "Head Turned toward the Left",
    "Head Turned toward the Right",
    "Patient Positioned Backward",
    "Patient Positioned Forward",
    "Slumped Neck Position",
    "Condyles Cut Off the Image",
    "Teeth Not Disoccluded",
    "Tongue Not Against the Roof of the Mouth",
    "Metal Artifact",
    "Patient’s Chin Not Against the Chin Rest",
    "Lead Apron Artifact",
    "Midline Shifted to the Left",
    "Midline Shifted to the Right",
    "Patient Movement",
    "No Obvious Error"
]
CLASSES = [s.lower() for s in CLASSES]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

# 2. 从文本中提取 <Error> ... </Error> 标签
def extract_errors(text):
    match = re.search(r"<Errors>(.*?)</Errors>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    content = match.group(1).strip()
    labels = [x.lower().split(":")[1].strip() for x in content.split(",") if x.strip()]
    return labels

# 3. 将类别文字转为 one-hot 向量
def labels_to_onehot(labels):
    vec = np.zeros(len(CLASSES), dtype=int)
    for lab in labels:
        if lab in CLASS2IDX:
            vec[CLASS2IDX[lab]] = 1
        else:
            print(f"[警告] 未知类别: {lab}")
    return vec

# 4. 读取数据
with open(gt_json, "r", encoding="utf-8") as f:
    gt_data = json.load(f)
with open(pred_json, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

y_true, y_pred = [], []
samples = []  # 保存每个样本的对比信息

for gt_item in gt_data:
    gt_img = gt_item["images"][0]
    # 提取 GT 标签
    gt_text = ""
    for m in gt_item["messages"]:
        if m["role"] == "assistant":
            gt_text = m["content"]
            break
    gt_labels = extract_errors(gt_text)
    gt_vec = labels_to_onehot(gt_labels)

    # 找到对应预测
    pred_item = next((p for p in pred_data if p["images"][0] == gt_img), None)
    if pred_item is None:
        print(f"[警告] 没找到预测: {gt_img}")
        continue
    pred_labels = extract_errors(pred_item["answer"])
    pred_vec = labels_to_onehot(pred_labels)

    y_true.append(gt_vec)
    y_pred.append(pred_vec)

    # 判断是否完全匹配
    # exact_match = int(np.array_equal(gt_vec, pred_vec))
    num_matcheds = np.sum(gt_vec & pred_vec)

    samples.append({
        "image": gt_img,
        "gt_labels": "; ".join(gt_labels) if gt_labels else "",
        "pred_labels": "; ".join(pred_labels) if pred_labels else "",
        "num_matcheds": num_matcheds
    })

y_true = np.array(y_true)
y_pred = np.array(y_pred)

metrics = compute_metrics(y_true, y_pred)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 5. 计算常见指标
print("=== 多标签分类评估指标 ===")
print("Hamming Loss:", hamming_loss(y_true, y_pred))
print("Micro Precision:", precision_score(y_true, y_pred, average="micro", zero_division=0))
print("Micro Recall:", recall_score(y_true, y_pred, average="micro", zero_division=0))
print("Micro F1:", f1_score(y_true, y_pred, average="micro", zero_division=0))
print("Macro Precision:", precision_score(y_true, y_pred, average="macro", zero_division=0))
print("Macro Recall:", recall_score(y_true, y_pred, average="macro", zero_division=0))
print("Macro F1:", f1_score(y_true, y_pred, average="macro", zero_division=0))

# 每个类别的详细报告
print("\n=== 每类指标 ===")
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

# 6. 保存结果到 CSV 文件
df = pd.DataFrame(samples)
df.to_csv("prediction_vs_gt_train.csv", index=False, encoding="utf-8-sig")
print("\n结果已保存到 prediction_vs_gt.csv")
