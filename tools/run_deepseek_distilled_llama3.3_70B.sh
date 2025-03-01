vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16/ --tensor-parallel-size 1 --max-model-len 32768 \
                                                        --enforce-eager \
                                                        --gpu-memory-utilization 0.99 \
                                                        --host 0.0.0.0 \
                                                        --port 8080 \
                                                        --served-model-name neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16
                                                        # --quantization gptq
