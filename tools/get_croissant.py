import requests
import json
API_URL = "https://huggingface.co/api/datasets/Bryceee/MM-Oral/croissant"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
with open('MMOral_Training_Dataset_croissant.json','w') as f:
    json.dump(data, f) 
print(data)
