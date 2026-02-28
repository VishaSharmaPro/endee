from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from endee import Endee

print("PDFs load ho rahi hain...")
loader = DirectoryLoader('medical_chatbot/data', glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
print(f" {len(documents)} pages load hue")

print(" Chunks ban rahe hain...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)
print(f" {len(text_chunks)} chunks bane")

print(" Embedding model load ho raha hai...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(" Endee se connect ho raha hai...")
client = Endee()

try:
    client.delete_index("medical_chatbot")
    print(" Purana index delete kiya")
except:
    pass

client.create_index(
    name="medical_chatbot",
    dimension=384,
    space_type="cosine",
    precision="float32"
)
print(" Index create hua")

index = client.get_index(name="medical_chatbot")

print(" Vectors store ho rahe hain...")
vectors_to_store = []
for i, chunk in enumerate(text_chunks):
    embedding = embedding_model.embed_query(chunk.page_content)
    vectors_to_store.append({
        "id": f"chunk_{i}",
        "vector": embedding,
        "meta": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "")
        }
    })

batch_size = 100
for i in range(0, len(vectors_to_store), batch_size):
    index.upsert(vectors_to_store[i:i+batch_size])
    print(f"  {min(i+batch_size, len(vectors_to_store))}/{len(vectors_to_store)} done...")

print(f"\n {len(vectors_to_store)} chunks Endee mein store ho gaye!")