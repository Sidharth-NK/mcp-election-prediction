import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

print("Listing available models...\n")

try:
    for model in client.models.list():
        print(f"- {model.name}")

except Exception as e:
    print(f"Error listing models: {e}")