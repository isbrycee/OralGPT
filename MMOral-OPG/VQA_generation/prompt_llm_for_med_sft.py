from openai import OpenAI
import gradio as gr
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

import re
from pprint import pprint
def parse_questions(input_text):
    # Split the text into Closed-End and Open-End sections based on keywords
    sections = input_text.split("Closed-End Questions")
    
    closed_end_text = sections[1].split("Open-End Questions")[0].strip() if len(sections) > 1 else ""
    open_end_text = sections[1].split("Open-End Questions")[1].strip() if "Open-End Questions" in sections[1] else ""

    # Helper function to extract question-answer pairs
    def parse_qa_string(input_str):
    # 定义正则表达式匹配question部分
        question_re = re.compile(
            r'\bquestion\b\s*:?\s*(.*?)(?=\s*\b(options|answer)\b)',
            re.IGNORECASE | re.DOTALL
        )
        question_match = question_re.search(input_str)
        if not question_match:
            raise ValueError("Question部分未找到")
        
        question = question_match.group(1).strip()
        next_keyword = question_match.group(2).lower() if question_match.group(2) else None
        remaining = input_str[question_match.end():]
        
        options = None
        # 处理Options部分
        if next_keyword == 'Options':
            options_re = re.compile(
                r'\boptions\b\s*:?\s*(.*?)(?=\s*\banswer\b)',
                re.IGNORECASE | re.DOTALL
            )
            options_match = options_re.search(remaining)
            if options_match:
                options = options_match.group(1).strip()
                remaining = remaining[options_match.end():]
            else:
                raise ValueError("Options格式错误")
        
        # 处理Answer部分
        answer_re = re.compile(r'\banswer\b\s*:?\s*(.*)', re.IGNORECASE | re.DOTALL)
        answer_match = answer_re.search(remaining)
        if not answer_match:
            raise ValueError("Answer部分未找到")
        answer = answer_match.group(1).strip()
        
        result = {
            'question': question,
            'answer': answer
        }
        if options is not None:
            result['options'] = options
        
        return result


    # Extract questions and answers for both sections
    closed_end_questions = parse_qa_string(closed_end_text)
    open_end_questions = parse_qa_string(open_end_text)

    # Return the parsed data
    return {
        "Closed-End Questions": closed_end_questions,
        "Open-End Questions": open_end_questions
    }


def batch_process_llm_requests(
    input_dir: str,
    output_dir: str,
    query_field: str = "query",
    response_field: str = "response",
    retries : int = 5
) -> None:
    """
    批量处理JSON文件，向LLM发送请求并保存结果到新文件夹
    
    :param input_dir: 输入JSON文件夹路径
    :param output_dir: 输出文件夹路径
    :param query_field: JSON文件中包含查询内容的字段名
    :param response_field: 要保存LLM响应的字段名
    """
    if retries < 1:
        print("尝试次数已用尽，无法转换为字典")
        return False

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
                llm_response = llm_response.split('\n</think>\n\n')[1]

                if "```json" in llm_response:
                    llm_response = llm_response.replace("```json", "").replace("```", "").strip()

                try:
                    _dict = json.loads(llm_response)
                    data[response_field] = _dict
                    # 保存到新路径
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)

                except json.JSONDecodeError as e:
                    print("解析失败，错误信息为:", e)
                    return batch_process_llm_requests(input_dir,output_dir,query_field,response_field,retries-1)
            
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


if __name__ == "__main__":
    # 输入输出路径配置
    input_dir = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report/"
    output_dir = "/home/jinghao/projects/x-ray-VLM/dataset/mmoral-json-v1/MM-Oral-OPG-jsons_latestv1_med_report_sft"
    
    # 运行批量处理
    batch_process_llm_requests(input_dir, output_dir, query_field="med_report", response_field="sft_data", retries=5)