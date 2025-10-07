# 🧠 VLMEvalKit 使用指南

> 本项目用于多模态大语言模型 (VLM) 的自动化评估与测试。  
> 请按照以下步骤完成环境配置与运行。

---

## 🚀 1. 环境配置

### 🧩 创建 `.env` 文件

在 `$VLMEvalKit/.env` 处创建并填写以下内容：
bash
OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
> 💡 `.env` 文件用于保存私密的 API 配置，请 **不要上传到公共仓库**！

---

## ⚙️ 2. 配置模型信息

打开并编辑 `vlmeval/config.py` 文件，示例如下：
python
注意：VLMEvalKit 使用的是 requests.post 方式，
因此需要使用 post 版本的 api_base
from functools import partial
from vlmeval.vlm import GPT4V
test_models = {
"gpt-4.1-nano": partial(
GPT4V,
model="gpt-4.1-nano",
api_base="https://www.dmxapi.cn/v1/chat/completions",
temperature=1,
img_size=-1,
img_detail="high",
retry=10,
verbose=True,
),
}
---

## 🔍 3. 检查模型配置是否成功

使用以下命令验证模型加载是否成功：
bash
vlmutil check gpt-4.1-nano
> ✅ 若返回模型可用或正常响应结果，即代表配置成功。

---

## ⚒️ 4. 配置运行参数

编辑或新建 `run_config.json` 文件，配置内容包括：

- 测试模型（例如上面的 `gpt-4.1-nano`）
- 测试数据集路径
- 评估方式及 Judger（如需）

示例结构：
json
{
"models": ["gpt-4.1-nano"],
"datasets": ["MMOral-Omni-Bench"],
"judger": "gpt-4o-mini",
"other_args": {}
}
---

## 🧭 5. 启动评估脚本
bash
python run.py --config run_config.json \
--mode all \
--api-nproc 4 \
--work-dir '.' \
--verbose
> 📌 如果想重复使用已有结果，可加上: `--reuse`

### 💬 参数说明：

| 参数 | 说明 |
|------|------|
| `--mode all` | 执行完整的评估流程 |
| `--api-nproc` | 设置并行请求数 |
| `--work-dir` | 指定工作目录 |
| `--verbose` | 显示详细日志 |

---

## ⚠️ 6. 使用注意事项

### 🗂️ 修改数据集文件

如需修改数据集文件，需同步更新以下内容：

- 文件：`MMOral-Omni-Bench.tsv`
- 对应的 MD5 值配置位于：  
  `$VLMEvalKit/vlmeval/dataset/image_vqa.py`  
  第 **1690** 行 与 第 **1694** 行

#### MD5 获取方式：
bash
md5sum file_path
---

### 🧹 可选：模型输出后处理

如需对模型输出结果进行后处理（例如去除 Think 部分，仅保留最终答案），请编辑文件：

`$VLMEvalKit/vlmeval/inference.py`

定位到第 **244** 行并修改相应逻辑即可。

---

## 💬 7. 反馈与贡献

若你在使用过程中遇到问题或有优化建议，欢迎通过以下方式反馈：

- 🐛 **提交 Issue**
- 💡 **提交 Pull Request**
- ⭐ **给仓库点个 Star 支持一下！**

## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!

If you use VLMEvalKit in your research or wish to refer to published OpenSource evaluation results, please use the following BibTeX entry and the BibTex entry corresponding to the specific VLM / benchmark you used.

```bib
@article{oralgpt2025,
  title={Towards Better Dental AI: A Multimodal Benchmark and Instruction Dataset for Panoramic X-ray Analysis},
  author={Hao, Jing and Fan, Yuxuan and Sun, Yanpeng and Guo, Kaixin and Lin, Lizhuo and Yang, Jinrong and Ai, Qi Yong H and Wong, Lun M and Tang, Hao and Hung, Kuo Feng},
  journal={arXiv preprint arXiv:2509.09254},
  year={2025}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>
