from openai import OpenAI
import gradio as gr
from examplar_data import example_input, example_output
from system_prompt import base_prompt
import os
import json
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1/"
model_name = "neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16"

# 生成最终系统提示
system_content = base_prompt
# system_content = base_prompt.format(
#     input_example=example_input,
#     output_example=example_output
# )

# 创建一个 OpenAI 客户端，用于与 API 服务器进行交互
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def batch_process_llm_requests(
    input_dir: str,
    output_dir: str,
    query_field: str = "query",
    response_field: str = "response"
) -> None:
    """
    批量处理JSON文件，向LLM发送请求并保存结果到新文件夹
    
    :param input_dir: 输入JSON文件夹路径
    :param output_dir: 输出文件夹路径
    :param query_field: JSON文件中包含查询内容的字段名
    :param response_field: 要保存LLM响应的字段名
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for filename in tqdm(json_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # 读取并处理JSON文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data: Dict[str, Any] = json.load(f)
                
                # 检查是否包含查询字段
                if query_field not in data:
                    raise ValueError(f"JSON文件 {filename} 缺少必需的 '{query_field}' 字段")
                
                # 调用LLM获取响应
                llm_response = get_llm_response(data[query_field])
                
                # 添加LLM响应到数据
                data[response_field] = llm_response.split('\n</think>\n\n')[1]
                
            # 保存到新路径
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {str(e)}")

def get_llm_response(query: str) -> str:
    """
    获取单个查询的LLM响应（非流式）
    
    :param query: 查询内容
    :return: LLM生成的响应
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    # history_openai_format = [{"role": "system", "content": f'{system_content}'}]
    # history_openai_format.append({"role": "user", "content": example_input})
    # history_openai_format.append({"role": "assistant", "content": example_output})
    # history_openai_format.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.98,
        stream=False,  # 使用非流式获取完整响应
        extra_body={
            'repetition_penalty': 1,
            'stop_token_ids': [7]
        }
    )
    
    return response.choices[0].message.content


def predict(message, history):
    # 将聊天历史转换为 OpenAI 格式
    history_openai_format = [{"role": "system", "content": f'{system_content}'}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    # 创建一个聊天完成请求，并将其发送到 API 服务器
    stream = client.chat.completions.create(
        model=model_name,   # 使用的模型名称
        messages= history_openai_format,  # 聊天历史
        temperature=1.0,                  # 控制生成文本的随机性
        stream=True,                      # 是否以流的形式接收响应
        extra_body={
            'repetition_penalty': 1, 
            'stop_token_ids': [7]
        }
    )

    # 从响应流中读取并返回生成的文本
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


# 在原有代码后添加以下内容（确保Gradio界面不会同时运行）
if __name__ == "__main__":
    # 输入输出路径配置
    input_dir = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_loc/"
    output_dir = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report/"
    
    # 运行批量处理
    batch_process_llm_requests(input_dir, output_dir, query_field="loc_caption", response_field="med_report")