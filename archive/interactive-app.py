from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 🔧 Disable cloud LLM usage
Settings.llm = None

# 📄 Load documents from the "data" folder
documents = SimpleDirectoryReader("data").load_data()

# 🤖 Use a local HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧠 Create index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 🧠 Query engine based on vector similarity
query_engine = index.as_query_engine()

print("✅ PDF loaded and indexed. Ask me anything! (type 'exit' to quit)\n")

# 🔁 Interactive loop
while True:
    query = input("❓ Your question: ")
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting. Have a great day!")
        break

    response = query_engine.query(query)
    print(f"💡 Answer: {response}\n")

