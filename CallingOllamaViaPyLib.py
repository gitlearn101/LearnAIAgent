import ollama

res = ollama.chat(
    model="llama3.2:latest",
    messages=[{"role": "user", "content": "why earth is round?"}],
)

#print(res)
print(res["message"]["content"]) # clean text output