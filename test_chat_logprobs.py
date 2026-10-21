
import ollama
import importlib.metadata

MODEL = "gemma4:latest"

print("Ollama Python package version:",
      importlib.metadata.version("ollama"))

print("\nTesting chat endpoint...")

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": (
                "Explain what a computer firewall does "
                "in three sentences. Use simple language."
            ),
        }
    ],
    options={"temperature": 0.0, "num_predict": 100},
    logprobs=True,
    top_logprobs=5,
    think=False,
)

data = response.model_dump()
message = data.get("message") or {}
records = data.get("logprobs") or []

print("\nGenerated response:")
print(message.get("content"))

print("\nMetadata:")
for key in ("done", "done_reason", "eval_count"):
    print(f"{key}: {data.get(key)}")

print("\nLogprob record count:", len(records))

for record in records[:10]:
    print(
        f"Token: {record.get('token')!r} | "
        f"Logprob: {record.get('logprob')}"
    )
