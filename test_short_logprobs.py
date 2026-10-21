
import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "gemma4:latest",
    "prompt": "Reply with exactly: Hello.",
    "stream": False,
    "think": False,
    "logprobs": True,
    "top_logprobs": 5,
    "options": {
        "temperature": 0,
        "num_predict": 10,
    },
}

response = requests.post(url, json=payload, timeout=300)
response.raise_for_status()
data = response.json()

print("Response:", repr(data.get("response")))
print("Done reason:", data.get("done_reason"))
print("Generated tokens:", data.get("eval_count"))

records = data.get("logprobs") or []
print("Logprob records:", len(records))

for record in records:
    print(record)
