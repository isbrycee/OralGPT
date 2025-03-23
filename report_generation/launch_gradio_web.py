from openai import OpenAI
import gradio as gr
from examplar_data import example_input, example_output
from system_prompt import base_prompt

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1/"
model_name = "neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16"

# 生成最终系统提示
system_content = base_prompt.format(
    input_example=example_input,
    output_example=example_output
)


# 创建一个 OpenAI 客户端，用于与 API 服务器进行交互
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def predict(message, history):
    # 将聊天历史转换为 OpenAI 格式
    history_openai_format = [{"role": "system", "content": f'{system_content}'}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    # 创建一个聊天完成请求，并将其发送到 API 服务器
    stream = client.chat.completions.create(
        model=model_name,   # 使用的模型名称
        messages= history_openai_format,  # 聊天历史
        temperature=0.8,                  # 控制生成文本的随机性
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

# 新增清除历史函数
def clear_history():
    return [], []  # 返回空列表来重置聊天历史和消息缓存

# 创建一个聊天界面，并启动它，share=True 让 gradio 为我们提供一个 debug 用的域名
gr.ChatInterface(predict).queue().launch(share=True)
