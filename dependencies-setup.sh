#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🍏 Detected macOS environment."
echo "🔧 Creating virtual environment..."

# Use python3 installed via Homebrew or system default
PYTHON_BIN=$(which python3)

# Create virtual environment
$PYTHON_BIN -m venv venv
source venv/bin/activate

echo "⬆️ Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing LlamaIndex + Ollama + HuggingFace dependencies..."
pip install \
  llama-index \
  llama-index-llms-ollama \
  llama-index-embeddings-huggingface \
  sentence-transformers

echo "✅ Setup complete."
echo "🚀 To activate your environment later, run: source venv/bin/activate"
echo "📌 Make sure Ollama is installed and running. Install from https://ollama.com"

