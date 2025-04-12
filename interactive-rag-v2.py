from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
import os

# Load local model
llm = Ollama(model="llama3")  # Or change to mistral or tinyllama if you prefer

Settings.llm = llm

# Load PDFs from directory
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Get query engine with source node metadata
query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

print("✅ PDF loaded and indexed. Ask me anything! (type 'exit' to quit)\n")

# Interactive loop
while True:
    query = input("❓ Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    response = query_engine.query(query)

    # Fuzzy logic: inspect source node scores
    if hasattr(response, "source_nodes"):
        relevant_nodes = [n for n in response.source_nodes if n.score and n.score >= 0.7]

        if not relevant_nodes:
            print("🤔 Sorry, I couldn't find an answer to that.\n")
            continue

    # Just show the answer, clean
    answer = response.response if hasattr(response, "response") else str(response)
    print(f"💡 Answer: {answer}\n")

