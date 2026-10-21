
import json
import ollama

response = ollama.generate(
    model="gemma4:latest",
    prompt="Explain what a computer firewall does in three sentences.",
    options={
        "temperature": 0.0,
        "num_predict": 120,
    },
    logprobs=True,
    top_logprobs=5,
    think=False,
)

# The SDK returns a Pydantic model, so convert it to a dictionary.
data = response.model_dump()

print("Response keys:")
print(list(data.keys()))

print("\nGenerated response:")
print(data.get("response"))

records = data.get("logprobs") or []

print("\nNumber of logprob records:")
print(len(records))

print("\nFirst five records:")
print(json.dumps(records[:5], indent=2, ensure_ascii=False, default=str))

print("\nCompletion metadata:")
for key in (
    "done",
    "done_reason",
    "prompt_eval_count",
    "eval_count",
    "eval_duration",
):
    if key in data:
        print(f"{key}: {data[key]}")
