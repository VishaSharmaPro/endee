import os
from dotenv import load_dotenv
from endee import Endee
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Endee connect
client = Endee()
index = client.get_index(name="medical_chatbot")

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# HuggingFace Inference Client
hf_client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN
)

print(" Medical Chatbot Ready! (exit likhke band karo)\n")

while True:
    user_query = input("Aap: ")
    if user_query.lower() == 'exit':
        break

    # Endee se similar chunks dhundo
    query_vector = embedding_model.embed_query(user_query)
    results = index.query(vector=query_vector, top_k=3)
    context = "\n\n".join([r["meta"]["text"] for r in results])

    # Prompt banao
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant. Answer only based on the given context. If you don't know, say 'I don't know'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {user_query}"
        }
    ]

    # LLM se answer lo
    response = hf_client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.5
    )

    answer = response.choices[0].message.content
    print(f"\nBot: {answer}\n")