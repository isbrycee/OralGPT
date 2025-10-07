# python run.py --data MMVet_Oral \
#               --model /home/jinghao/projects/llm/LLaMA-Factory/saves/qwen2_vl-7b/lora/report_qa_chat/checkpoint-127649 \
#               --mode all \
#               --api-nproc 4 \
#               --work-dir '.' \
#               --verbose

python run.py --config run_config.json \
              --api-nproc 8 \
              --work-dir '.' \
              --verbose \
              --mode all \
              --reuse

