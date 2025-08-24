import os
import glob
from openai import OpenAI
import base64

# 初始化客户端（确保已设置 OPENAI_API_KEY 环境变量）

client = OpenAI(
    api_key="sk-xxx",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)

# 视频帧所在的文件夹路径
frames_folder = "Vident-real/train/Vident-real-demo/GT"

# 获取所有 png 图片，按文件名排序
frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))

# 这里可以选择“抽帧”避免太多图片，比如每隔10帧取1张
selected_frames = frame_files[::100]

# 构造输入消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are analyzing a real intra-oral surgical video recorded during conservative dental treatment. The video captures complex clinical conditions inside the oral cavity, where the scene is crowded with multiple dental instruments and artifacts. It includes conditions such as occlusions, frequent appearance variations, tool–tooth interactions, bleeding, water spray, splashing fluids, motion blur, strong light reflections, and occasional camera fouling. The footage also shows non-standard tools, intra-oral mirrors, and other interfering objects that partially obstruct the view. Please generate a detailed, structured description of the intra-oral surgical procedure shown in the video, highlighting the clinical environment, the types of tools and tissues visible, and the dynamic interactions occurring during treatment."},
        ]
    }
]

# 把本地图片转成 base64，然后作为 data URI 传给模型
def encode_image_to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"url": f"data:image/png;base64,{b64}"}

for frame in selected_frames:
    messages[0]["content"].append({
        "type": "image_url",
        "image_url": encode_image_to_data_uri(frame)
    })


# 调用 GPT-4o 进行视频帧理解
response = client.chat.completions.create(
    model="GLM-4.1V-9B-Thinking",  # 或 gpt-4o
    messages=messages,
)

# 输出结果
print(response.choices[0].message.content)