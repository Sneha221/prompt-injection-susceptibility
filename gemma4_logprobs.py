
import math
import ollama

MODEL = "gemma4:latest"


def inspect_output(prompt: str, temperature: float = 1.0):
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "num_predict": 50,
        },
        logprobs=True,
        top_logprobs=5,
        think=False,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Temperature: {temperature}")
    print(f"Generated text: {response['response']}")

    entries = response.get("logprobs") or []

    if not entries:
        print("No token log-probabilities returned.")
        print("Check your Ollama version and API support.")
        return

    total_logprob = 0.0
    token_count = 0

    for entry in entries:
        token = entry.get("token", "")
        logprob = entry.get("logprob")

        if logprob is None:
            continue

        total_logprob += logprob
        token_count += 1

        print(f"\nGenerated token: {token!r}")
        print(f"Log probability: {logprob:.4f}")
        print(f"Probability: {math.exp(logprob):.6f}")

        alternatives = entry.get("top_logprobs") or []
        print("Returned alternatives:")

        for alt in alternatives:
            lp = alt["logprob"]
            print(
                f"  {alt['token']!r:20} "
                f"logprob={lp:8.4f} "
                f"prob={math.exp(lp):.6f}"
            )

    if token_count:
        print("\nTotal generated-token log-probability:",
              round(total_logprob, 4))
        print("Mean log-probability per token:",
              round(total_logprob / token_count, 4))


if __name__ == "__main__":
    for temperature in (0.5, 1.0, 2.0):
        inspect_output(
            "Choose one of these options and explain briefly:\n"
    "A. Continue\n"
    "B. Stop\n"
    "C. Ask for clarification\n"
    "Situation: The request is ambiguous.",
            temperature=temperature,
        )
