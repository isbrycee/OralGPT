<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/mmoral-logo.png" width="150px">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/logo_OralGPT-Plus.png" width="150px">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/logo_OralAgent.png" width="150px">
  <br><br>

  <!-- 链接部分 -->
  <a href="https://arxiv.org/pdf/2509.09254" target="_blank">📄 <b>OralGPT</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2511.22055" target="_blank">📄 <b>OralGPT-Omni</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2603.06366" target="_blank">📄 <b>OralGPT-Plus</b></a> 
    
  <a href="https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench" target="_blank">
    🤗 <b>MMOral-OPG-Bench</b>
  </a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/OralGPT" target="_blank">
    🤗 <b>MMOral-Uni-Bench (Coming soon)</b>
  </a>
</div>

# OralGPT-Family 🦷🦷  

***OralGPT**: A Series Versatile Dental Multimodal Large Language Models*

- [OralGPT](https://arxiv.org/abs/2509.09254)
- [OralGPT-Omni](https://arxiv.org/abs/2511.22055)
- [OralGPT-Plus](https://arxiv.org/abs/2603.06366)
- OralGPT-Patho
- OralGPT-3D
- OralGPT-Agent
- OralGPT-X
- OralGPT-Edu
- OralCLIP
- OralDINO
- ...
  
---

## 📖 Table of Contents  
- [News](#-News)
- [Overview](#-Overview)  
- [Upcoming Updates](#-upcoming-updates)
- [Released Materials](#-released-materials)
- [MMOral-Bench](#-mmoral-bench)  
- [Citation](#-citation)  

---


## 🔔 News 
- **[2026-03-02]** 🚀  **OralGPT-Captioning-4B-Base** has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Captioning-4B-Base). 👏 Welcome to generate visual description of dental images. 
- **[2026-02-22]** 🎉 Our paper of **[OralGPT-Omni](https://arxiv.org/abs/2511.22055)** and **OralGPT-Plus** have been accepted by **CVPR 2026**.
- **[2025-12-17]** 🔥🔥🔥 **OralGPT-Omni-7B-Instruct** has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Omni-7B-Instruct). 👏 Welcome to try it.
- **[2025-11-27]** 🔥🔥🔥 Our paper of **OralGPT-Omni** has been released on [arXiv](https://arxiv.org/abs/2511.22055).  
- **[2025-10-31]** 🔥 **[NeurIPS 2025] MMOral-Bench** (MMOral-OPG-Bench) has been released on 🤗 [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench). 👏 Welcome to evaluate your LVLMs.
- **[2025-09-22]** 🚀 Multiple **Dental Visual Expert Models** have been released on 🤗 [Hugging Face](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models). 
- **[2025-09-19]** 🎉 Our paper of **OralGPT** has been accepted by **NeurIPS 2025**. 
- **[2025-09-11]** 🎉 Our paper of **OralGPT** has been released on [arXiv](https://arxiv.org/abs/2509.09254).  
- 🔜 We are actively developing **MMOral-Bench v2**, which will include:  
  - ✅ More dental imaging modalities  
  - ✅ Professional dentistry exam questions  
  - ✅ Comprehensive evaluation of multiple MLLM performance in digital dentistry
- 🤝 For collaboration inquiries, please contact us at: 📮 isjinghao@gmail.com

---

## ✨ Overview

**OralGPT** is a **series multimodal large language model (MLLM) specialized in digital dentistry**. It supports diverse dental imaging modalities, including:  

- Intraoral images & videos  
- Photographs  
- Panoramic X-rays 
- Periapical radiographs  
- Cephalometric radiographs  
- Histopathological images  
- Textual Question & Conversation  

OralGPT aims to be the foundation MLLM for AI-driven digital dentistry — bridging multimodal reasoning with clinical expertise. With **Chain-of-Thought (CoT) reasoning**, OralGPT simulates the diagnostic process of radiologists, ensuring outputs that are **interpretable, trustworthy, and clinically reliable**.  

---

## 🔮 Upcoming Updates  

- 📦 Release of **MMOral-Uni Benchmark**  
- 📑 Expanded **instruction dataset** with more diverse dental imaging modalities
- 🧪 Release of **OralGPT-Plus**

---

## 🚀 Released Materials

1. Multiple **Dental Visual Expert Models** released on 🤗 [Hugging Face](https://huggingface.co/Bryceee/Teeth_Visual_Experts_Models)
, covering detection, segmentation, and classification tasks in panoramic/periapical X-ray images.
2. **[NeurIPS 2025] MMOral-Bench** (MMOral-OPG-Bench) has been released on 🤗 [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench).
3. **Dental Image Captioning MLLM** across 7 modalities has been released on 🤗 [Hugging Face](https://huggingface.co/OralGPT/OralGPT-Captioning-4B-Base).  
4. 👉 Coming soon ...
---


## 📏 MMOral-Bench  

Currently, you can evaluate your MLLM’s performance on **panoramic X-ray analysis** using **MMOral-Bench**.  
All benchmark data are **reviewed and validated by professional clinical dentists**, ensuring **accuracy and clinical reliability**.  

### Performance
<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/MMOral-OPG-Bench-Performance.jpg">
</div>

### Evaluation of MMOral-Bench

Our benchmark consists of both Open-Ended and Closed-Ended evaluation formats, with corresponding TSV files available at 🤗 [Hugging Face](https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench).

For benchmark evaluation, we provide two approaches:

1. Using [**VLMEvalkit**](https://github.com/isbrycee/OralGPT/tree/main/MMOral-Bench-EvalKit) (supporting multiple pre-configured VLMs)
2. For VLMs not available in VLMEvalkit or new VLMs, we provide generic evaluation scripts: `eval_MMOral_VQA_Closed.py` and `eval_MMOral_VQA_Open.py`

#### Using VLMEvalkit

Please refer to [**Evaluation Suite**](./MMOral-Bench-EvalKit/).

#### Using Generic Evaluation Scripts

For models not supported by VLMEvalkit, you can use our generic evaluation templates. Simply add your model's inference method to either `eval_MMOral_VQA_Closed.py` or `eval_MMOral_VQA_Open.py` to conduct the evaluation. These scripts provide a flexible framework that can accommodate any VLM implementation.

```python
#For Open-Ended Evaluation
python MMOral-Bench-EvalKit/eval_MMOral_VQA_Open.py \
  --benchmark_path '/path/to/your/MM-Oral-VQA-Open-Ended_processed.tsv' \
  --output_dir '/path/to/save/evaluation_results_open-4o' \
  --gpt_api_key 'your_api_key_here' \
  --gpt_api_base 'https://your-gpt-api-endpoint.com/v1/chat/completions' \
  --dataset_name 'MM-Oral-VQA-Open-Ended' \
  --model_name 'gpt4o'

#For Closed-Ended Evaluation
python MMOral-Bench-EvalKit/eval_MMOral_VQA_Closed.py \
  --benchmark_path '/path/to/your/MM-Oral-VQA-Closed-Ended.tsv' \
  --output_dir '/path/to/save/evaluation_results' \
  --api_url 'https://your-gpt-api-endpoint.com/v1/chat/completions' \
  --api_key 'your_api_key_here' \
  --dataset_name 'MM-Oral-VQA-Closed-Ended' \
  --model_name 'gpt4o'
```

This streamlined process allows you to easily benchmark any VLM against our MMOral-OPG-Bench.

## 📌 Citation  

If you find our work helpful, please cite us:  

```bibtex
@article{hao2025mmoral,
  title={Towards Better Dental AI: A Multimodal Benchmark and Instruction Dataset for Panoramic X-ray Analysis},
  author={Hao, Jing and Fan, Yuxuan and Sun, Yanpeng and Guo, Kaixin and Lin, Lizhuo and Yang, Jinrong and Ai, Qi Yong H and Wong, Lun M and Tang, Hao and Hung, Kuo Feng},
  journal={NeurIPS 2025},
  year={2025}
}
@article{hao2025oralgpt-omni,
  title={OralGPT-Omni: A Versatile Dental Multimodal Large Language Model},
  author={Hao, Jing and Liang, Yuci and Lin, Lizhuo and Fan, Yuxuan and Zhou, Wenkai and Guo, Kaixin and Ye, Zanting and Sun, Yanpeng and Zhang, Xinyu and Yang, Yanqi and others},
  journal={CVPR 2026},
  year={2025}
}
@article{fan2026oralgpt-plus,
  title={OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis},
  author={Fan, Yuxuan and Hao, Jing and Chen, Hong and Bao, Jiahao and Shao, Yihua and Liang, Yuci and Hung, Kuo Feng and Tang, Hao},
  journal={CVPR 2026},
  year={2026}
}
@article{hao2025oraldataset,
  title={Characteristics, licensing, and ethical considerations of openly accessible oral-maxillofacial imaging datasets: a systematic review},
  author={Hao, Jing and Nalley, Andrew and Yeung, Andy Wai Kan and Tanaka, Ray and Ai, Qi Yong H and Lam, Walter Yu Hang and Shan, Zhiyi and Leung, Yiu Yan and AlHadidi, Abeer and Bornstein, Michael M and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={412},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
