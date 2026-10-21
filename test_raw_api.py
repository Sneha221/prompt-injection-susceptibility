
import requests

BASE_URL = "http://localhost:11434"

# 1. Check the Ollama server version
try:
    version_response = requests.get(
        f"{BASE_URL}/api/version",
        timeout=10,
    )
    version_response.raise_for_status()
    print("Ollama server version:")
    print(version_response.json())

except requests.RequestException as error:
    print("Could not retrieve server version:", error)
    raise SystemExit(1)


# 2. Call the API directly, bypassing the Ollama Python SDK
payload = {
    "model": "gemma4:latest",
    "prompt": (
        "Explain what a computer firewall does in three sentences. "
        "Use simple language."
    ),
    "stream": False,
    "think": False,
    "logprobs": True,
    "top_logprobs": 5,
    "options": {
        "temperature": 0.0,
        "num_predict": 100,
    },
}

response = requests.post(
    f"{BASE_URL}/api/generate",
    json=payload,
    timeout=300,
)

print("\nHTTP status:", response.status_code)
response.raise_for_status()

data = response.json()
records = data.get("logprobs") or []

print("\nGenerated text:")
print(data.get("response"))

print("\nGenerated-token count:", data.get("eval_count"))
print("Raw API logprob record count:", len(records))

print("\nFirst five logprob records:")
for record in records[:5]:
    print({
        "token": record.get("token"),
        "logprob": record.get("logprob"),
    })

print("\nResponse JSON keys:")
print(list(data.keys()))
