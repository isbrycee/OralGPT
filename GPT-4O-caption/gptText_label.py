import requests
import os
import json

def chat_with_gpt(prompt):
    api_key = "# input your api_key "
    model = "gpt-4o"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an oral photography visual assistant with professional and rich knowledge in dentistry, and you are looking at a human "
                           "oral photography. What you see are provided with a brief text depicting the whole image together"
                           " with several regions of visual information describing the same image you are looking at. "
                           "In the pictures of oral photography you see a total of {num_boxes} teeth. They are located at positions [(point_x1, point_y1), (point_x2, point_y2)] etc. in the image."
                           "These coordinates represent the central points corresponding to the teeth in the image, "
                           "the floating point number ranges from 0 to 1. "
                           "Remember to return the given character information directly without any modification and insert"
                           "the character information into the generated detailed caption. Based on the image you see and these "
                           "descriptions I provided, integrate them to generate a more comprehensive, accurate, fluent, and "
                           "detailed description of the image. The generated description must include the total number of teeth, "
                           " and the normalized coordinate point position. Note that you must "
                           "output the normalized coordinates of each tooth and the number of tooth cannot make any guesses. Also "
                           "note that you must be extremely sensitive and cautious about the number of teeth, and do not subjectively "
                           "change the data provided to you about the number of teeth."

            },
            {
                "role": "user",
                "content":
                    "What you see is an oral photography of a human mouth, which contains 14 teeth. "
                    "The coordinates of these teeth are located at [(0.53, 0.79), (0.44, 0.79), (0.62, 0.78), (0.35, 0.77), "
                    "(0.71,0.74), (0.27, 0.72), (0.77, 0.66), (0.21,0.64), (0.81, 0.57), (0.16,0.56), (0.87, 0.46), (0.11,0.45), "
                    "(0.91, 0.32), (0.07,0.28)] in the image."

            },
            {
                "role": "assistant",
                "content": "This is an oral photography of 14 teeth. You can see the teeth are located respectively in the image at: "
                           "[(0.53, 0.79), (0.44, 0.79),(0.62, 0.78), (0.35, 0.77), (0.71,0.74), (0.27, 0.72), (0.77, 0.66), (0.21,0.64)),"
                           "(0.81, 0.57), (0.16,0.56), (0.87, 0.46), (0.11,0.45), (0.91, 0.32), (0.07,0.28)]."
            },
            {
                "role": "user",
                "content":
                    "What you see is an oral photography of a human mouth, which contains 6 teeth. "
                    "The coordinates of these teeth are located at [(0.70, 0.56), (0.78,0.53), (0.23, 0.59), (0.83, 0.54)"
                    ", (0.17,0.57), (0.88, 0.46)] in the image."
            },
            {
                "role": "assistant",
                "content": "What you see is an oral photography of a human mouth, which contains 6 teeth. "
                           "The position of these teeth in the image is [(0.59,0.62), (0.78,0.53), (0.23, 0.59), (0.83, 0.54), (0.17,0.57), "
                           "(0.88, 0.46)]."
            },
            {
                "role": "user",
                "content": prompt
            }

        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    print("waiting response")
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

def format_description(txt_content):
    lines = txt_content.strip().splitlines()
    total_teeth = len(lines)
    positions = []

    # Iterate through each line to extract positions
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "0":  # Only process lines with category '0'
            x, y = round(float(parts[1]), 2), round(float(parts[2]), 2)
            positions.append(f"({x}, {y})")

    # Construct the description
    description = f"This is an oral photograph showing a total of {total_teeth} teeth. "
    if positions:
        positions_str = ", ".join(positions)
        description += f"The coordinates of these teeth are located at {positions_str} in the image."

    return description



def process_files_in_folder(folder_path, max_files=5):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')][
                :max_files]  # Only select the first `max_files` files

    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            txt_content = file.read()
            # Format the content of the txt file into descriptive text
            formatted_prompt = format_description(txt_content)
            # Send the generated description to chat_with_gpt function
            response = chat_with_gpt(formatted_prompt)
            print(f"Response for {file_name}:\n{response}\n")


process_files_in_folder("E:\oral agent\chagpt\example\label_1", max_files=5)

# response = chat_with_gpt(prompt)
# print(response)
