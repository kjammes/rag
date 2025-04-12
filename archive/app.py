from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

Settings.llm = None

# Load documents locally from a directory
documents = SimpleDirectoryReader("data").load_data()

# Use a local HuggingFace model for embeddings (no cloud dependency)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a VectorStore index using the local embedding model
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Persist the index locally
index.storage_context.persist(persist_dir="./index_storage")

# Query engine setup, no cloud models involved
query_engine = index.as_query_engine()

# Perform a local query (RAG-style)
response = query_engine.query("Summarize the document")
print(response)

