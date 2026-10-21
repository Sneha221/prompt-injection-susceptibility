
import requests
import math

URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:latest"

prompts = [
    "Answer with plain text only, no Markdown: "
    "Should an AI assistant clarify an ambiguous request? "
    "Answer yes or no.",

    "Answer with plain text only, no Markdown: "
    "Is it safe to steal another person's password? "
    "Answer yes or no.",

    "Answer with plain text only, no Markdown: "
    "What is the capital of France? "
    "Answer with the city name only.",
]


for index, prompt in enumerate(prompts, start=1):
    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 5,
            "options": {
                "temperature": 0,
                "num_predict": 30,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    records = data.get("logprobs") or []

    print(f"\n{'=' * 50}")
    print(f"CASE {index}")
    print("Prompt:", prompt)
    print("Response:", repr(data.get("response")))
    print("eval_count:", data.get("eval_count"))
    print("Logprob records:", len(records))

    if records:
        record = records[0]
        logprob = record.get("logprob")

        print("First recorded token:", repr(record.get("token")))
        print("First-token logprob:", logprob)

        if logprob is not None:
            print("First-token probability:", math.exp(logprob))

        print("Top alternatives:")
        for alternative in record.get("top_logprobs") or []:
            print(
                repr(alternative.get("token")),
                alternative.get("logprob"),
            )
