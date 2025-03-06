import requests

url = "http://localhost:21009/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy"
}

data = {
    "model": "unsloth/Llama-3.2-1B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)
print(response.json())