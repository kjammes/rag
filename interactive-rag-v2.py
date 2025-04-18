from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
import os

# Load local model (e.g., Ollama)
llm = Ollama(model="llama3")  # Or change to mistral or tinyllama if you prefer

# Create embeddings using HuggingFace (you can switch to other embeddings if required)
# Specify to use a local model
embedding = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")  # Ensure it's a local model

# Set up the sentence splitter (to split large texts into manageable chunks)
sentence_splitter = SentenceSplitter()

# Set the LLM and embedding in the Settings (instead of using ServiceContext)
Settings.llm = llm
Settings.embedding = embedding
Settings.node_parser = sentence_splitter
Settings.embed_model = embedding

# Load PDFs from directory
documents = SimpleDirectoryReader("./data").load_data()

# Create an index from the loaded documents using the settings
index = VectorStoreIndex.from_documents(documents)

# Get query engine with source node metadata (set top_k to control how many similar nodes to retrieve)
query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

# Start querying
print("✅ PDFs loaded and indexed. Ask me anything! (type 'exit' to quit)")

while True:
    query = input("❓ Your question: ")
    if query.lower() == 'exit':
        break

    # Get the response for the query from the index
    response = query_engine.query(query)

    # Print the response
    print("💡 Answer:", response)

