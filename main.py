
#from langchain_community.chat_models import ChatOllama  # ✅ use Ollama instead of OpenAI
from langchain_ollama import ChatOllama

# initialize Ollama model (make sure ollama serve is running and mistral is pulled)
llm = ChatOllama(model="mistral")


# ask something
response = llm.invoke("What is Apollo 13?")

print("\n--- Ollama Response ---\n")

print(response.content)   # ✅ .content contains the text

print("\n\n--- Done ---")



""" using ChatGPT Or Claude
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is apollo 13?")
print(response)
"""