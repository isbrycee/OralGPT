import os
import json
import time
import signal
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
from vllm import LLM, SamplingParams

# 如果需要导入custom modules，请确保路径正确
try:
    from examplar_data import example_input, example_output
    from system_prompt import base_prompt
except ImportError:
    # 如果导入失败，提供默认值
    base_prompt = "You are a helpful assistant."
    example_input = ""
    example_output = ""


class ModelRunner:
    def __init__(self, model_name, tensor_parallel_size=4, max_model_len=32768, 
                 dtype="float16", gpu_memory_utilization=0.99):
        """初始化模型运行器"""
        self.model_name = model_name
        self.system_content = base_prompt
        
        print(f"正在加载模型: {model_name} (使用 {dtype} 精度)")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,  # 使用fp16加速推理
        )
        print("模型加载完成")
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.98,
            top_p=0.95,
            repetition_penalty=1.0,
            max_tokens=1024,
        )

    def get_llm_response(self, query):
        """获取单个查询的LLM响应"""
        # 构建完整的提示
        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": query}
        ]
        
        # 将消息列表转换为单个提示字符串
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|system|>\n{msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n"
        
        prompt += "<|assistant|>\n"  # 添加助手回应的开始标记
        
        # 生成响应
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text

    def batch_process_llm_requests(self, input_dir, output_dir, query_field="query", 
                                   response_field="response", batch_size=4,
                                   shard_id=0, num_shards=1):
        """
        批量处理JSON文件
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径
            query_field: JSON中查询字段名
            response_field: 要存储响应的字段名
            batch_size: 每批处理的文件数
            shard_id: 当前分片ID (从0开始)
            num_shards: 总分片数
        """
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 检查点文件路径
        ckpt_dir = './'
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        ckpt_file = os.path.join(ckpt_dir, f"checkpoint_shard_{shard_id}.json")
        
        # 获取所有JSON文件
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        json_files.sort()  # 排序确保分片结果一致
        
        # 根据分片参数选择要处理的文件
        total_files = len(json_files)
        files_per_shard = total_files // num_shards
        remainder = total_files % num_shards
        
        # 计算当前分片的开始和结束索引
        start_idx = shard_id * files_per_shard + min(shard_id, remainder)
        end_idx = start_idx + files_per_shard + (1 if shard_id < remainder else 0)
        
        # 获取当前分片需要处理的文件
        shard_files = json_files[start_idx:end_idx]
        
        # 处理检查点
        processed_files = set()
        last_processed_index = 0
        if os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, 'r', encoding='utf-8') as f:
                    ckpt_data = json.load(f)
                    processed_files = set(ckpt_data.get("processed_files", []))
                    last_processed_index = ckpt_data.get("last_index", 0)
                    print(f"从检查点恢复: 已处理 {len(processed_files)} 文件，上次索引位置: {last_processed_index}")
            except Exception as e:
                print(f"读取检查点文件时出错: {str(e)}，将从头开始处理")
        
        # 过滤掉已处理的文件
        files_to_process = [f for f in shard_files if f not in processed_files]
        
        print(f"分片 {shard_id+1}/{num_shards}: 处理 {len(files_to_process)}/{len(shard_files)} 个文件 (总计: {total_files})")
        
        # 从上次处理的索引开始
        current_files = files_to_process
        
        # 将文件分批
        batches = [current_files[i:i + batch_size] for i in range(0, len(current_files), batch_size)]
        
        processed_count = len(processed_files)
        with tqdm(total=len(shard_files), initial=processed_count, desc=f"分片 {shard_id+1}/{num_shards} 处理文件") as pbar:
            for batch_idx, batch in enumerate(batches):
                # 读取每个批次的数据
                batch_data = []
                batch_prompts = []
                
                for filename in batch:
                    input_path = os.path.join(input_dir, filename)
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if query_field not in data:
                                print(f"警告: 文件 {filename} 缺少必需的 '{query_field}' 字段")
                                continue
                            
                            # 构建提示
                            prompt = f"<|system|>\n{self.system_content}\n<|user|>\n{data[query_field]}\n<|assistant|>\n"
                            batch_data.append((filename, data))
                            batch_prompts.append(prompt)
                    except Exception as e:
                        print(f"读取文件 {filename} 时发生错误: {str(e)}")
                
                # 没有有效数据，跳过此批次
                if not batch_data:
                    continue
                
                try:
                    # 批量生成响应
                    outputs = self.llm.generate(batch_prompts, sampling_params=self.sampling_params)
                    
                    # 处理响应并保存结果
                    batch_processed = []
                    for i, (filename, data) in enumerate(batch_data):
                        if i < len(outputs):
                            response_text = outputs[i].outputs[0].text
                            
                            # 提取think标记后的内容（如果存在）
                            if "\n</think>\n\n" in response_text:
                                response_text = response_text.split('\n</think>\n\n')[1]
                                
                            # 添加到数据
                            data[response_field] = response_text
                            
                            # 保存到输出文件
                            output_path = os.path.join(output_dir, filename)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False, indent=4)
                            
                            batch_processed.append(filename)
                    
                    # 更新处理进度
                    processed_files.update(batch_processed)
                    processed_count += len(batch_processed)
                    pbar.update(len(batch_processed))
                    
                    # 更新检查点 (每批次更新)
                    current_index = batch_idx * batch_size + len(batch_processed)
                    with open(ckpt_file, 'w', encoding='utf-8') as f:
                        ckpt_data = {
                            "processed_files": list(processed_files),
                            "last_index": current_index,
                            "last_update": time.time(),
                            "total_processed": len(processed_files)
                        }
                        json.dump(ckpt_data, f, ensure_ascii=False, indent=4)
                    
                except Exception as e:
                    print(f"处理批次时发生错误: {str(e)}")
                    # 即使出错也保存检查点
                    with open(ckpt_file, 'w', encoding='utf-8') as f:
                        ckpt_data = {
                            "processed_files": list(processed_files),
                            "last_index": batch_idx * batch_size,
                            "last_update": time.time(),
                            "total_processed": len(processed_files),
                            "error": str(e)
                        }
                        json.dump(ckpt_data, f, ensure_ascii=False, indent=4)
        
        print(f"分片 {shard_id+1}/{num_shards} 处理完成: 成功处理 {processed_count}/{len(shard_files)} 文件")
        
        # 完成后更新检查点
        with open(ckpt_file, 'w', encoding='utf-8') as f:
            ckpt_data = {
                "processed_files": list(processed_files),
                "last_index": len(shard_files),
                "last_update": time.time(),
                "total_processed": len(processed_files),
                "completed": True
            }
            json.dump(ckpt_data, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="使用vLLM处理JSON文件批量任务")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
                        help="要加载的模型名称或路径")
    parser.add_argument("--tp-size", type=int, default=4, 
                        help="张量并行大小（使用的GPU数量）")
    parser.add_argument("--max-model-len", type=int, default=32768, 
                        help="模型最大上下文长度")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"],
                        help="模型精度")
    parser.add_argument("--gpu-mem-util", type=float, default=0.99, 
                        help="GPU内存利用率")
    
    # 数据参数
    parser.add_argument("--input-dir", type=str, 
                        default="../unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc/",
                        help="输入JSON文件所在目录")
    parser.add_argument("--output-dir", type=str, 
                        default="../unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc_wreport/",
                        help="输出JSON文件目录")
    parser.add_argument("--query-field", type=str, default="loc_caption",
                        help="JSON中的查询字段名")
    parser.add_argument("--response-field", type=str, default="med_report",
                        help="存储响应的字段名")
    
    # 处理参数
    parser.add_argument("--batch-size", type=int, default=4,
                        help="每批处理的文件数量")
    parser.add_argument("--shard-id", type=int, default=1,
                        help="当前分片ID (从0开始)")
    parser.add_argument("--num-shards", type=int, default=4,
                        help="总分片数")
    
    # 检查点参数
    parser.add_argument("--ignore-checkpoint", action="store_true",
                        help="忽略现有检查点并从头开始")
    
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 如果指定了忽略检查点，删除现有检查点文件
    if args.ignore_checkpoint:
        ckpt_file = os.path.join('./', f"checkpoint_shard_{args.shard_id}.json")
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
            print(f"已删除检查点文件: {ckpt_file}")
    
    # 创建模型运行器
    runner = ModelRunner(
        model_name=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util
    )
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n接收到中断信号，正在安全退出...")
        print("程序将在当前批次完成后退出，检查点已保存")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 运行批处理
        runner.batch_process_llm_requests(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            query_field=args.query_field, 
            response_field=args.response_field,
            batch_size=args.batch_size,
            shard_id=args.shard_id,
            num_shards=args.num_shards
        )
        
    except KeyboardInterrupt:
        print("程序已安全中断")
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        # 在异常情况下也保存检查点
        ckpt_file = os.path.join('./', f"checkpoint_shard_{args.shard_id}.json")
        with open(ckpt_file, 'a', encoding='utf-8') as f:
            f.write(f"\n# 错误信息: {str(e)}\n")


if __name__ == "__main__":
    main()
