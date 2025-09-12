# OralGPT 👄🦷  
*An Omni Multimodal Large Language Model for Digital Dentistry*

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![MLLM](https://img.shields.io/badge/MLLM-OralGPT-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-MMOral-red.svg)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-MMOral--Bench-orange.svg)]()

<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/mmoral-logo.png" width="150px">
</div>

---

## 📖 Table of Contents  
- [Introduction](#-introduction)  
- [News](#-News)
- [Upcoming Updates](#-upcoming-updates)  
- [MMOral-Bench](#-mmoral-bench)  
- [Citation](#-citation)  

---

## ✨ Introduction

**OralGPT** is a **series multimodal large language model (MLLM) specialized in digital dentistry**. It supports diverse dental imaging modalities, including:  

- Intraoral images & videos  
- Photographs  
- Panoramic X-rays 
- Periapical radiographs  
- Cephalometric radiographs  
- Histopathological slides  
- Textual Question & Conversation  

OralGPT aims to be the foundation MLLM for AI-driven digital dentistry — bridging multimodal reasoning with clinical expertise. With **Chain-of-Thought (CoT) reasoning**, OralGPT simulates the diagnostic process of radiologists, ensuring outputs that are **interpretable, trustworthy, and clinically reliable**.  

---

## 🔔 News 

- **[2025-09-11]** 🎉 Our paper of **OralGPT** has been released on [arXiv](https://arxiv.org/abs/2509.09254).  
- 🔜 We are actively developing **MMOral-Bench v2**, which will include:  
  - ✅ More dental imaging modalities  
  - ✅ Professional dentistry exam questions  
  - ✅ Comprehensive evaluation of multiple MLLM performance in digital dentistry
- 🤝 For collaboration inquiries, please contact us at: 📮 isjinghao@gmail.com

---

## 🔮 Upcoming Updates  

- 📦 Release of **MMOral-Bench v2**  
- 📑 Expanded **instruction dataset** with more diverse dental imaging modalities
- 🧪 Release of **OralGPT-O3**

---

## 📏 MMOral-Bench  

Currently, you can evaluate your MLLM’s performance on **panoramic X-ray analysis** using **MMOral-Bench**.  
All benchmark data are **reviewed and validated by professional clinical dentists**, ensuring **accuracy and clinical reliability**.  

### Performance

| Model | Close-ended VQA (Teeth) | Patho | His | Jaw | Summ | Overall | Open-ended VQA (Teeth) | Patho | His | Jaw | Summ | Report | Overall | Avg. |
|-------|--------------------------|-------|-----|-----|------|---------|-------------------------|-------|-----|-----|------|--------|---------|------|
| **Proprietary LVLMs** ||||||||||||||| 
| GPT-4o-2024-11-20 [21] | 29.65 | 40.99 | 46.71 | 55.81 | 56.25 | 45.40 | 31.48 | 26.05 | 37.56 | 57.42 | 30.37 | 42.50 | 37.50 | 41.45 |
| GPT-4V [21] | 37.88 | 39.13 | 48.50 | 51.69 | 55.83 | 46.00 | 27.16 | 13.47 | 33.50 | 58.95 | 30.84 | 45.00 | 34.83 | 39.12 |
| Claude-3.7-Sonnet-20250219 [1] | 49.31 | 44.28 | 47.31 | 56.00 | 51.69 | 50.72 | 36.93 | 26.65 | 42.39 | 51.09 | 28.04 | 50.00 | 39.67 | 41.04 |
| Gemini-2.5-Flash-preview-04-17 [33] | 20.80 | 16.00 | 16.15 | 27.51 | 10.42 | 22.00 | 35.99 | 22.76 | 40.61 | 51.53 | 32.71 | 45.50 | 39.20 | 30.54 |
| Gemini-2.0-Flash [33] | 37.17 | 35.40 | 44.61 | 45.34 | 46.89 | 41.20 | 37.27 | 26.05 | 40.36 | 52.40 | 35.05 | 49.00 | 40.67 | 40.94 |
| Qwen-Max-VL-2025-04-08 [5] | 18.41 | 11.18 | 27.55 | 32.96 | 47.92 | 22.00 | 10.22 | 7.30 | 11.12 | 22.88 | 6.86 | 27.00 | 14.33 | 18.15 |
| Step-1o-turbo [1] | 31.86 | 24.22 | 39.52 | 38.92 | 55.81 | 41.67 | 33.02 | 21.56 | 30.20 | 51.31 | 31.31 | 45.00 | 36.00 | 36.00 |
| Doubao-1-5-thinking-vision-pro-250428 [2] | 28.20 | 27.33 | 23.35 | 19.85 | 31.25 | 24.80 | 34.38 | 25.45 | 35.90 | 56.11 | 39.72 | 49.00 | 40.33 | 32.57 |
| **Open-Source LVLMs** ||||||||||||||| 
| Deepseek-VL-7b-chat [27] | 22.65 | 17.39 | 28.74 | 59.93 | 52.08 | 31.20 | 12.75 | 8.16 | 8.40 | 30.00 | 13.14 | 9.10 | 13.42 | 22.31 |
| Emu3-chat [40] | 40.89 | 44.72 | 37.73 | 60.67 | 43.75 | 45.80 | 18.02 | 7.02 | 15.50 | 28.53 | 12.44 | 9.60 | 16.05 | 30.93 |
| Qwen2.5-VL-72B [6] | 26.55 | 27.95 | 26.35 | 22.47 | 47.92 | 26.80 | 13.05 | 18.44 | 11.66 | 26.88 | 7.44 | 11.50 | 14.77 | 20.79 |
| CogVLM2-19B [39] | 40.63 | 31.68 | 34.13 | 38.95 | 60.42 | 35.20 | 26.11 | 17.09 | 26.86 | 49.24 | 18.14 | 24.50 | 27.63 | 31.42 |
| GLM-4V-9B [15] | 29.03 | 45.40 | 41.36 | 62.55 | 64.58 | 40.20 | 17.85 | 8.01 | 17.46 | 41.22 | 15.93 | 19.40 | 21.38 | 30.85 |
| LLaVA-NeXT-13B-hf [26] | 32.92 | 32.92 | 30.54 | 38.20 | 60.42 | 33.80 | 14.48 | 10.28 | 9.23 | 22.41 | 14.30 | 21.30 | 15.43 | 24.62 |
| LLaVA-OneVision [22] | 40.51 | 40.51 | 32.08 | 49.30 | 71.25 | 44.20 | 22.68 | 18.43 | 21.63 | 32.49 | 15.72 | 21.00 | 20.93 | 27.62 |
| LLaMA-3.2-Vision-11B-Instruct [16] | 27.90 | 21.12 | 40.42 | 53.40 | 60.42 | 34.40 | 28.01 | 20.71 | 25.03 | 33.65 | 17.56 | 19.50 | 23.97 | 28.99 |
| Cambrian-34B [37] | 46.87 | 44.36 | 44.31 | 70.04 | 60.42 | 44.40 | 33.10 | 21.42 | 31.83 | 48.24 | 13.30 | 16.00 | 29.63 | 37.02 |
| Phi-4-multimodal-instruct [3] | 36.28 | 36.65 | 49.03 | 53.00 | 43.75 | 40.20 | 20.82 | 11.47 | 27.69 | 23.49 | 13.92 | 12.80 | 20.54 | 30.37 |
| InternVL3-38B [9] | 29.67 | 21.12 | 25.75 | 39.33 | 31.25 | 28.40 | 33.69 | 22.41 | 29.70 | 46.11 | 20.23 | 42.90 | 34.15 | 31.82 |
| Chameleon-7B [28] | 31.62 | 32.75 | 25.63 | 33.93 | 42.50 | 32.00 | 8.05 | 6.02 | 9.63 | 9.15 | 3.35 | 8.40 | 7.27 | 21.54 |
| PaliGemma-3B [7] | 26.02 | 24.22 | 42.52 | 47.57 | 35.42 | 33.80 | 8.20 | 9.65 | 8.70 | 13.29 | 6.16 | 0.60 | 7.78 | 20.49 |
| MiniCPM-02.6 [43] | 27.46 | 45.45 | 40.20 | 43.53 | 49.58 | 37.40 | 29.20 | 17.38 | 24.38 | 46.75 | 15.35 | 27.90 | 28.42 | 32.91 |
| Kosmos-2 [32] | 15.75 | 18.01 | 28.14 | 10.11 | 25.00 | 17.40 | 13.58 | 10.71 | 11.18 | 19.76 | 8.49 | 3.40 | 11.87 | 14.64 |
| mPLUG-Owl3-7B [45] | 34.16 | 32.30 | 36.53 | 71.91 | 62.50 | 42.80 | 12.50 | 8.44 | 8.52 | 30.50 | 3.26 | 19.30 | 13.22 | 28.94 |
| Gemma3-12B [34] | 31.78 | 35.61 | 32.15 | 45.34 | 50.00 | 35.60 | 26.39 | 21.00 | 20.63 | 26.88 | 23.33 | 33.20 | 25.32 | 29.65 |
| XComposer2-VL-7B [50] | 26.49 | 26.71 | 21.56 | 38.06 | 45.83 | 23.20 | 6.52 | 11.02 | 15.00 | 7.67 | 2.10 | 8.53 | 8.99 | 16.10 |
| Molmo-72B-0924 [11] | 28.26 | 14.91 | 29.94 | 40.29 | 22.92 | 25.60 | 9.25 | 6.31 | 3.49 | 16.25 | 5.00 | 9.28 | 8.43 | 17.01 |
| Yi-VL-34B [47] | 48.81 | 36.64 | 41.31 | 42.00 | 70.83 | 42.00 | 24.97 | 23.40 | 20.59 | 39.35 | 15.23 | 9.90 | 22.98 | 31.59 |
| Qwen-QVQ-72B [36] | 48.67 | 49.07 | 59.28 | 74.53 | 72.92 | 56.60 | 30.72 | 12.28 | 16.75 | 41.05 | 22.09 | 24.5 | 22.78 | 39.58 |
| Ovis2-34B [29] | 45.84 | 51.55 | 53.89 | 79.40 | 79.17 | 56.80 | 32.48 | 24.33 | 31.60 | 50.88 | 21.05 | 31.70 | 33.02 | **44.91** |
| Kimi-VL-A3B-Thinking [35] | 25.84 | 27.33 | 25.75 | 29.96 | 27.08 | 26.80 | 52.53 | 37.66 | 53.79 | 68.59 | 50.93 | 61.50 | 54.55 | 40.68 |
| **Medical Specific LVLMs** ||||||||||||||| 
| LLaVA-Med [23] | 25.49 | 26.71 | 21.56 | 13.86 | 45.83 | 23.20 | 23.23 | 18.75 | 11.56 | 32.82 | 26.28 | 5.30 | 19.21 | 21.40 |
| HuatuoGPT-V-34B [8] | 41.86 | 14.91 | 29.94 | 56.29 | 22.92 | 32.00 | 32.69 | 12.65 | 28.05 | 36.52 | 13.80 | 15.00 | 29.48 | 27.54 |
| HealthGPT-XL32 [25] | 48.27 | 45.14 | 51.50 | 76.41 | 79.17 | 52.20 | 29.80 | 22.16 | 11.41 | 47.82 | 24.77 | 10.00 | 27.17 | **39.59** |
| MedVLM-R1 [31] | 32.19 | 31.68 | 37.72 | 65.17 | 47.92 | 38.60 | 22.58 | 15.22 | 21.57 | 40.61 | 21.96 | 24.50 | 24.58 | 31.59 |
| MedDr [19] | 36.46 | 36.02 | 41.92 | 73.03 | 64.58 | 46.00 | 27.50 | 28.14 | 30.20 | 49.17 | 26.17 | 7.50 | 26.17 | 36.09 |

### Evaluation of MMOral-Bench

Our benchmark consists of both Open-Ended and Closed-Ended evaluation formats, with corresponding TSV files available at [https://huggingface.co/datasets/EasonFan/MMOral-Bench](https://huggingface.co/datasets/EasonFan/MMOral-Bench).

For benchmark evaluation, we provide two approaches:

1. Using [**VLMEvalkit**](https://github.com/open-compass/VLMEvalKit) (supporting multiple pre-configured VLMs)
2. For VLMs not available in VLMEvalkit or new VLMs, we provide generic evaluation scripts: `eval_MMOral_VQA_Closed.py` and `eval_MMOral_VQA_Open.py`

#### Using VLMEvalkit

We offer a zip file which includes the version we use to evaluate the VLMs on the MMOral-Bench. We have included an `mmoral.py` file in the `vlmeval/dataset` directory. To evaluate any model supported by VLMEvalkit:

1. Modify the `mmoral_config.json` file with your desired settings
2. Run `bash run_eval.sh` to start the evaluation

#### Using Generic Evaluation Scripts

For models not supported by VLMEvalkit, you can use our generic evaluation templates. Simply add your model's inference method to either `eval_MMOral_VQA_Closed.py` or `eval_MMOral_VQA_Open.py` to conduct the evaluation. These scripts provide a flexible framework that can accommodate any VLM implementation.

```python
#For Open-Ended Evaluation
python MMOral-Bench-Eval/eval_MMOral_VQA_Open.py \
  --benchmark_path '/path/to/your/MM-Oral-VQA-Open-Ended_processed.tsv' \
  --output_dir '/path/to/save/evaluation_results_open-4o' \
  --gpt_api_key 'your_api_key_here' \
  --gpt_api_base 'https://your-gpt-api-endpoint.com/v1/chat/completions' \
  --dataset_name 'MM-Oral-VQA-Open-Ended' \
  --model_name 'gpt4o'

#For Closed-Ended Evaluation
python MMOral-Bench-Eval/eval_MMOral_VQA_Closed.py \
  --benchmark_path '/path/to/your/MM-Oral-VQA-Closed-Ended.tsv' \
  --output_dir '/path/to/save/evaluation_results' \
  --api_url 'https://your-gpt-api-endpoint.com/v1/chat/completions' \
  --api_key 'your_api_key_here' \
  --dataset_name 'MM-Oral-VQA-Closed-Ended' \
  --model_name 'gpt4o'
```

This streamlined process allows you to easily benchmark any VLM against our MMOral-Bench dataset.

## 📌 Citation  

If you find our work helpful, please cite us:  

```bibtex
@article{oralgpt2025,
  title   = {Towards Better Dental AI: A Multimodal Benchmark and Instruction Dataset for Panoramic X-ray Analysis},
  author  = {Jing Hao, Yuxuan Fan, Yanpeng Sun, Kaixin Guo, Lizhuo Lin, Jinrong Yang, Qi Yong H. Ai, Lun M. Wong, Hao Tang, Kuo Feng Hung},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.09254}
}
