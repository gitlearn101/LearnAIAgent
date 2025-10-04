import requests
import json

url = "http://localhost:11434/api/generate"

payload = {
    "model": "mistral",
    "prompt": "Write a short haiku about autumn"
}

# We stream so the response prints live
response = requests.post(url, json=payload, stream=True)

print("\n--- Ollama Response ---\n")

full_response = ""

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        chunk = data.get("response", "")
        full_response += chunk
        print(chunk, end="", flush=True)  # force text to appear immediately

print("\n\n--- Done ---")

