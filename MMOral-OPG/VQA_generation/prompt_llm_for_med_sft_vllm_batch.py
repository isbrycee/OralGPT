from vllm import LLM, SamplingParams
from system_prompt import base_prompt
import os
import json
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import re
import signal
import time

# Configuration
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
system_content = base_prompt

class ModelRunner:
    def __init__(self, model_name, tensor_parallel_size=4, max_model_len=32768, 
                 dtype="bfloat16", gpu_memory_utilization=0.99):
        """Initialize the model runner"""
        self.model_name = model_name
        self.system_content = system_content
        
        print(f"Loading model: {model_name} (using {dtype} precision)")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )
        print("Model loaded successfully")
        
        # Set sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.0,
            max_tokens=12288,
        )

    def get_llm_response(self, query):
        """Get LLM response for a single query"""
        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": query}
        ]
        
        # Convert messages to a single prompt string
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|system|>\n{msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n"
        
        prompt += "<|assistant|>\n"  # Add assistant response start marker
        
        # Generate response
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text

    def batch_process_llm_requests(
        self,
        input_dir: str,
        output_dir: str,
        query_field: str = "med_report",
        response_field: str = "sft_data",
        batch_size: int = 8,  # Increased batch size
        shard_id: int = 0,
        num_shards: int = 1,
        exclude_fields: List[str] = ['properties', 'loc_caption', 'med_report'],
    ) -> None:
        """
        Batch process JSON files, send requests to LLM and save results to new folder
        
        Args:
            input_dir: Input JSON folder path
            output_dir: Output folder path
            query_field: Field name in JSON file containing query content
            response_field: Field name to save LLM response
            batch_size: Number of files to process in each batch
            shard_id: Current shard ID (starting from 0)
            num_shards: Total number of shards
            exclude_fields: Fields to exclude when saving output JSON
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file path
        ckpt_dir = './'
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        ckpt_file = os.path.join(ckpt_dir, f"checkpoint_shard_{shard_id}.json")
        
        # Get all JSON files
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        json_files.sort()  # Sort to ensure consistent sharding
        
        # Select files based on shard parameters
        total_files = len(json_files)
        files_per_shard = total_files // num_shards
        remainder = total_files % num_shards
        
        # Calculate start and end indices for current shard
        start_idx = shard_id * files_per_shard + min(shard_id, remainder)
        end_idx = start_idx + files_per_shard + (1 if shard_id < remainder else 0)
        
        # Get files for current shard
        shard_files = json_files[start_idx:end_idx]
        
        # Process checkpoint
        processed_files = set()
        last_processed_index = 0
        if os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, 'r', encoding='utf-8') as f:
                    ckpt_data = json.load(f)
                    processed_files = set(ckpt_data.get("processed_files", []))
                    last_processed_index = ckpt_data.get("last_index", 0)
                    print(f"Resuming from checkpoint: Processed {len(processed_files)} files, last index: {last_processed_index}")
            except Exception as e:
                print(f"Error reading checkpoint file: {str(e)}, starting from scratch")
        
        # Filter out already processed files
        files_to_process = [f for f in shard_files if f not in processed_files]
        
        print(f"Shard {shard_id+1}/{num_shards}: Processing {len(files_to_process)}/{len(shard_files)} files (total: {total_files})")
        
        # Split files into batches
        batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]
        
        processed_count = len(processed_files)
        with tqdm(total=len(shard_files), initial=processed_count, desc=f"Shard {shard_id+1}/{num_shards} processing files") as pbar:
            for batch_idx, batch in enumerate(batches):
                # Read data for each batch
                batch_data = []
                batch_prompts = []
                batch_filenames = []
                batch_original_data = []
                
                for filename in batch:
                    input_path = os.path.join(input_dir, filename)
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if query_field not in data:
                                print(f"Warning: File {filename} missing required '{query_field}' field")
                                continue
                            
                            # Build prompt
                            prompt = f"<|system|>\n{self.system_content}\n<|user|>\n{data[query_field]}\n<|assistant|>\n"
                            batch_data.append((filename, data))
                            batch_original_data.append(data)
                            batch_prompts.append(prompt)
                            batch_filenames.append(filename)
                    except Exception as e:
                        print(f"Error reading file {filename}: {str(e)}")
                
                # Skip batch if no valid data
                if not batch_data:
                    continue
                
                try:
                    # Generate responses in batch
                    outputs = self.llm.generate(batch_prompts, sampling_params=self.sampling_params)
                    
                    # Process responses and save results
                    batch_processed = []
                    for i, ((filename, original_data), data) in enumerate(zip(batch_data, batch_original_data)):
                        if i < len(outputs):
                            max_retries = 5
                            retry_count = 0
                            success = False
                            
                            while not success and retry_count < max_retries:
                                response_text = outputs[i].outputs[0].text
                                
                                # Extract content after think tag if exists
                                if "</think>" in response_text:
                                    split_text = response_text.split('</think>')
                                    response_text = split_text[1].strip()
                                
                                # Try to parse JSON response
                                if "```json" in response_text:
                                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                                
                                try:
                                    parsed_json = json.loads(response_text)
                                    
                                    # Create a new data dict excluding the specified fields
                                    filtered_data = {k: v for k, v in original_data.items() if k not in exclude_fields}
                                    filtered_data[response_field] = parsed_json
                                    
                                    success = True
                                except json.JSONDecodeError:
                                    # If cannot parse JSON, retry
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        print(f"JSON parse failed, retrying {filename} (attempt {retry_count})")
                                        # Regenerate response
                                        retry_outputs = self.llm.generate([batch_prompts[i]], sampling_params=self.sampling_params)
                                        outputs[i] = retry_outputs[0]
                                    else:
                                        print(f"Max retries reached, saving raw response: {filename}")
                                        # Create filtered data with raw text
                                        filtered_data = {k: v for k, v in original_data.items() if k not in exclude_fields}
                                        filtered_data[response_field] = response_text
                                        success = True
                                
                                if success:
                                    # Save to output file
                                    output_path = os.path.join(output_dir, filename)
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
                                    
                                    batch_processed.append(filename)
                    
                    # Update progress
                    processed_files.update(batch_processed)
                    processed_count += len(batch_processed)
                    pbar.update(len(batch_processed))
                    
                    # Update checkpoint after each batch
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
                    print(f"Error processing batch: {str(e)}")
                    # Save checkpoint even on error
                    with open(ckpt_file, 'w', encoding='utf-8') as f:
                        ckpt_data = {
                            "processed_files": list(processed_files),
                            "last_index": batch_idx * batch_size,
                            "last_update": time.time(),
                            "total_processed": len(processed_files),
                            "error": str(e)
                        }
                        json.dump(ckpt_data, f, ensure_ascii=False, indent=4)
        
        print(f"Shard {shard_id+1}/{num_shards} completed: Successfully processed {processed_count}/{len(shard_files)} files")
        
        # Update checkpoint on completion
        with open(ckpt_file, 'w', encoding='utf-8') as f:
            ckpt_data = {
                "processed_files": list(processed_files),
                "last_index": len(shard_files),
                "last_update": time.time(),
                "total_processed": len(processed_files),
                "completed": True
            }
            json.dump(ckpt_data, f, ensure_ascii=False, indent=4)


def setup_signal_handlers():
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, exiting safely...")
        print("Program will exit after current batch completes, checkpoint saved")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Input/output path configuration
    input_dir = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons_latestv3_wloc_wreport"
    output_dir = "/hpc2hdd/home/yfan546/workplace/xray_teeth/unlabeled_data/MM-Oral-OPG-jsons_report_sft"
    
    # Setup signal handlers for safe interruption
    setup_signal_handlers()
    
    # Fields to exclude in output files
    exclude_fields = ['properties', 'loc_caption', 'med_report']
    
    # Create model runner with optimized parameters
    runner = ModelRunner(
        model_name=model_name,
        tensor_parallel_size=4,  # Increased for more parallelism
        max_model_len=32768,
        dtype="bfloat16",
        gpu_memory_utilization=0.99,
    )
    
    try:
        # Run batch processing with larger batch size
        runner.batch_process_llm_requests(
            input_dir=input_dir,
            output_dir=output_dir,
            query_field="med_report",
            response_field="sft_data",
            batch_size=8,  # Increased batch size
            shard_id=0,
            num_shards=2,
            exclude_fields=exclude_fields
        )
    except KeyboardInterrupt:
        print("Program safely interrupted")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
