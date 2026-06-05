import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
chat_model = os.getenv("GEMINI_CHAT_MODEL")
embed_model = os.getenv("GEMINI_EMBEDDING_MODEL")

print(f"Key loaded: {api_key[:8]}...")
print(f"Chat model: {chat_model}")
print(f"Embed model: {embed_model}\n")

genai.configure(api_key=api_key)

print("Testing chat...")
try:
    model = genai.GenerativeModel(chat_model)
    response = model.generate_content("Say hello in one word")
    print(f"SUCCESS: {response.text.strip()}\n")
except Exception as e:
    print(f"ERROR: {e}\n")

print("Testing embeddings...")
try:
    result = genai.embed_content(
        model=embed_model,
        content="test sentence",
        task_type="retrieval_document"
    )
    print(f"SUCCESS: vector dimension = {len(result['embedding'])}")
except Exception as e:
    print(f"ERROR: {e}")