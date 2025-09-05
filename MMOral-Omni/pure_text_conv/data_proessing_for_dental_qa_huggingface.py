import os
import json
from openai import OpenAI
import asyncio
from tqdm.asyncio import tqdm  # 注意用异步版本的 tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# 创建一个线程池
executor = ThreadPoolExecutor(max_workers=15)  # 增加线程池的线程数

client = OpenAI(
    api_key="sk-N1hsISExwkdoyisZg9gTd8CxzNAwK8r2ESRSbFsp2M2859Q6",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)

# 系统提示，可以根据需要修改
SYSTEM_PROMPT = "You are a professional assistant for analyzing medical forum conversations."

# 用户任务提示（英文版）
USER_PROMPT = """
You will receive a JSON object with the following fields:
- "title": a short summary of the dialogue
- "dialogue": a text conversation between a patient and a doctor on a forum

Your tasks:
1. Carefully read and understand the dialogue based on meaning, not just punctuation like colons.
2. Extract the patient’s entire description as "user instruction".
3. Extract the doctor's entire description as "model response". Hint: the "Dr xxx said" is usually about the doctor's response.
4. If there are multiple turns, do not merge them. Instead, preserve them in the "conversations" array as alternating pairs:
   - Patient’s turns must appear at odd positions (1st, 3rd, 5th, …) as `"from": "human"`.
   - Doctor’s turns must appear at even positions (2nd, 4th, 6th, …) as `"from": "gpt"`.
   - Ensure the dialogue is structured strictly as patient → doctor → patient → doctor, and so on.
5. Determine whether the dialogue relies on additional input beyond text, such as images. If the patient’s dialogue includes any mention of images, or if the doctor mentions that they found relevant information from an image, then discard the dialogue and output the string "Invalid dialogue".
6. IMPORTANT: Do NOT include usernames, quotes, prefixes (like "nowicki2023:", "Dr M said:", "Click to expand...", etc.), or any forum‑specific formatting. Only output the plain dialogue content as if the patient and doctor were directly talking to each other.
7. The final output must strictly follow this JSON structure:

[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      },
      ...
    ],
    "title": "",
    "system": "system prompt (optional)"
  }
]

Do not include explanations, comments, or any extra text — output only the JSON in the required format.
"""


# def process_file(file_path, max_try=5, sleep_sec=1):
#     """调用大模型处理一个文件，其中 data 是一个包含多个 dict 的 list"""
#     with open(file_path, "r", encoding="utf-8") as f:
#         data_list = json.load(f)  # data_list 是一个 list

#     results = []

#     for item in tqdm(data_list[:20]):
#         title = item.get("title", "")
#         dialogue = item.get("dialogue", "")

#         attempt = 0
#         parsed_result = None

#         while attempt < max_try and parsed_result is None:
#             attempt += 1
#             try:
#                 response = client.chat.completions.create(
#                     model="glm-4.5-free",  # 你可以换成 gpt-4.1/gpt-4o 等
#                     messages=[
#                         {"role": "system", "content": SYSTEM_PROMPT},
#                         {"role": "user", "content": USER_PROMPT},
#                         {"role": "user", "content": json.dumps(
#                             {"title": title, "dialogue": dialogue}, 
#                             ensure_ascii=False
#                         )}
#                     ],
#                     temperature=0.1,
#                 )

#                 result_text = response.choices[0].message.content.strip()

#                 print("Model response:", result_text)  # 打印模型的原始输出，方便调试
#                 if result_text:
#                     if result_text == "Invalid dialogue":
#                         # 如果模型返回的是 "Invalid dialogue"，则跳过该对话
#                         print("Skipping invalid dialogue.")
#                         break
#                     parsed_result = json.loads(result_text)
#                 else:
#                     time.sleep(sleep_sec)

#             except json.JSONDecodeError:
#                 # 解码失败，继续尝试
#                 time.sleep(sleep_sec)
#             except Exception as e:
#                 # 其他错误也允许重试
#                 time.sleep(sleep_sec)

#         if parsed_result is not None:
#             if result_text != "Invalid dialogue" and len(parsed_result[0]['conversations']) >=2:
#                 results.append(parsed_result)

#     return results  # 返回一个 list

# def process_folder(folder_path, output_path="merged_output.json"):
#     results = []
#     for fname in os.listdir(folder_path):
#         if fname.endswith(".json"):
#             file_path = os.path.join(folder_path, fname)
#             print(f"Processing {file_path}...")
#             result = process_file(file_path)
#             if result:
#                 # result 本身是一个数组，所以要 extend
#                 results.extend(result)

#             with open(os.path.join(folder_path, "sharGPT_data/", fname), "w", encoding="utf-8") as f:
#                 json.dump(result, f, indent=2, ensure_ascii=False)

    # print(f"处理完成，结果已保存到 {output_path}")

async def call_api(title, dialogue, max_try=5, sleep_sec=1):
    """异步调用大模型 API，带重试"""
    attempt = 0
    parsed_result = None

    while attempt < max_try and parsed_result is None:
        attempt += 1
        try:
            print(f"Calling API for title: {title}, attempt {attempt}/{max_try}...")
            # 记录开始时间
            start_time = time.time()
            # 使用线程池包装同步调用
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"title": title, "dialogue": dialogue},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    temperature=0.1,
                ),
            )
            # 记录结束时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"API call completed in {elapsed_time:.2f} seconds for length dialogue {len(dialogue)}")
            result_text = response.choices[0].message.content.strip()
            if result_text:
                if result_text == "Invalid dialogue":
                    print(f"Skipping invalid dialogue for title: {title}")
                    return None
                parsed_result = json.loads(result_text)
            else:
                print(f"No result returned for title: {title}, retrying...")
                await asyncio.sleep(sleep_sec)

        except json.JSONDecodeError as e:
            print(f"JSON decode error for title: {title}: {e}, retrying...")
            await asyncio.sleep(sleep_sec)
        except Exception as e:
            print(f"Error during API call for title: {title}: {e}, retrying...")
            await asyncio.sleep(sleep_sec)

    if parsed_result:
        print(f"Successfully processed title: {title}")
    else:
        print(f"Failed to process title: {title} after {max_try} attempts")

    return parsed_result if parsed_result and len(parsed_result[0].get("conversations", [])) >= 2 else None


async def process_file(file_path, max_concurrency=10, batch_size=10):
    """
    并发处理一个文件
    - max_concurrency: 控制最大同时并发数
    - batch_size: 每批处理多少条数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async def task(item):
        async with semaphore:
            title = item.get("title", "")
            dialogue = item.get("dialogue", "")
            return await call_api(title, dialogue)

    # 分批处理
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i : i + batch_size]
        tasks = [asyncio.create_task(task(item)) for item in batch]

        # 用 tqdm 进度条显示当前批次进度
        batch_number = i // batch_size + 1
        total_batches = (len(data_list) + batch_size - 1) // batch_size  # 计算总批次数
        with tqdm(total=len(tasks), desc=f"Processing Batch {batch_number}/{total_batches}") as pbar:
            for fut in asyncio.as_completed(tasks):
                try:
                    r = await fut
                    if r:
                        results.extend(r)
                except Exception as e:
                    print(f"任务执行出错: {e}")
                finally:
                    pbar.update(1)  # 每完成一个任务，进度条+1

    return results


async def process_folder(folder_path, output_path="merged_output.json"):
    results = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".json"):
            file_path = os.path.join(folder_path, fname)
            print(f"Processing {file_path}...")
            file_results = await process_file(file_path)
            if file_results:
                results.extend(file_results)

            # 输出到单个文件
            save_dir = os.path.join(folder_path, "sharGPT_data")
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, fname), "w", encoding="utf-8") as f:
                json.dump(file_results, f, indent=2, ensure_ascii=False)

    # # 如果需要合并所有结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    folder = "/home/jinghao/projects/x-ray-VLM/RGB/pure_text_conv_data/dental_QA_huggingface"  # 修改为你的文件夹路径
    asyncio.run(process_folder(folder))
