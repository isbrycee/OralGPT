# OralGPT 👄🦷  
*A Multimodal Large Language Model for Digital Dentistry*

<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/mmoral-logo.png" width="150px">
</div>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-MMOral-red.svg)]()
[![Benchmark](https://img.shields.io/badge/Benchmark-MMOral--Bench-orange.svg)]()

---

## 📖 Table of Contents  
- [Introduction](#-introduction)  
- [Features](#-features)  
- [MMOral Dataset](#-mmoral-dataset)  
- [MMOral-Bench](#-mmoral-bench)  
- [Upcoming Updates](#-upcoming-updates)  
- [Citation](#-citation)  
- [Contact](#-contact)  

---

## ✨ Introduction

**OralGPT** is a **series multimodal large language model (MLLM) specialized in digital dentistry**.  
It supports diverse dental imaging modalities, including:  

- Intraoral images & videos  
- Clinical photographs  
- Panoramic X-rays 
- Periapical radiographs  
- Cephalometric radiographs  
- Histopathological slides  
- Textual Question & Conversation  

With **Chain-of-Thought (CoT) reasoning**, OralGPT simulates the diagnostic process of radiologists, ensuring outputs that are **interpretable, trustworthy, and clinically reliable**.  

---

## 🔔 News 
🏃‍♀️ More datasets, models, and evaluations are under development. For collaboration, please contact: 📮 isjinghao@gmail.com

---

## 🔮 Upcoming Updates  

- 📦 Release of **MMOral-Bench v2**  
- 📑 Expanded **instruction dataset**  
- 🧪 Evaluation toolkit for reproducible benchmarking

---

## 📏 MMOral-Bench  

You can evaluate your MLLM’s performance on **panoramic X-ray analysis** using **MMOral-Bench**.  

🔜 We are actively developing **MMOral-Bench v2**, which will include:  
- ✅ More dental imaging modalities  
- ✅ Professional dentistry exam questions  
- ✅ Comprehensive evaluation of MLLM performance in **digital dentistry**  

All benchmark data are **reviewed and validated by senior clinical dentists**, ensuring **accuracy and clinical reliability**.  

---

## Evaluation of MMOral-Bench

Our benchmark consists of both Open-Ended and Closed-Ended evaluation formats, with corresponding TSV files available at [https://huggingface.co/datasets/EasonFan/MMOral-Bench](https://huggingface.co/datasets/EasonFan/MMOral-Bench).

For benchmark evaluation, we provide two approaches:

1. Using [**VLMEvalkit**](https://github.com/open-compass/VLMEvalKit) (supporting multiple pre-configured VLMs)
2. For VLMs not available in VLMEvalkit or new VLMs, we provide generic evaluation scripts: `eval_MMOral_VQA_Closed.py` and `eval_MMOral_VQA_Open.py`

### Using VLMEvalkit

We offer a zip file which includes the version we use to evaluate the VLMs on the MMOral-Bench. We have included an `mmoral.py` file in the `vlmeval/dataset` directory. To evaluate any model supported by VLMEvalkit:

1. Modify the `mmoral_config.json` file with your desired settings
2. Run `bash run_eval.sh` to start the evaluation

### Using Generic Evaluation Scripts

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
