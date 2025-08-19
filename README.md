# OralGPT

<div align="center">
  <img src="https://raw.githubusercontent.com/isbrycee/OralGPT/main/assets/mmoral-logo.png" width="200px">
</div>

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
