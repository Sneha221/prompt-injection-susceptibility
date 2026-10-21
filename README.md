# prompt-injection-susceptibility
Exploratory evaluation of LLM behavior and generation-time signals based on log probabilities
Token log-probability analysis
The project investigates whether generation-time token log probabilities from Ollama can provide useful signals for examining model responses.
The scripts:
- Request log probabilities and top-token alternatives through Ollama’s Python SDK and HTTP API.
- Inspect the first generated token’s log probability and convert it to a probability with exp(logprob).
- In detector.py, attempt to calculate response-level features such as mean log probability, mean surprisal, minimum log probability, and the fraction of tokens below a fixed exploratory threshold.
- Save selected token metadata alongside prompt-injection and safety-prompt results.
This is an exploration of available signals, not a working or validated safety detector. In the collected CSVs, the log-probability record count is usually one, so the current datasets do not reliably support response-level statistics across all generated tokens. First-token probability also measures only the model’s probability for that token in context; it is not a calibrated score for the safety or correctness of the full response.
