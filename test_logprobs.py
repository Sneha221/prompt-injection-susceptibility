
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma4:latest",
        "prompt": "Should an AI assistant ask for clarification when a request is ambiguous? Answer briefly.",
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {
            "temperature": 0,
            "num_predict": 80,
        },
    },
    timeout=300,
)

response.raise_for_status()
data = response.json()

records = data.get("logprobs") or []

print("Response:", data.get("response"))
print("Generated-token count:", data.get("eval_count"))
print("Logprob records:", len(records))

for record in records:
    print("\nRecorded token:", repr(record.get("token")))
    print("Logprob:", record.get("logprob"))
    print("Alternative tokens:")

    for alternative in record.get("top_logprobs") or []:
        print(
            repr(alternative.get("token")),
            alternative.get("logprob"),
        )
