#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/ai-env}"

echo "Starting AI Agent SMC setup..."
echo "Project directory: $PROJECT_DIR"
echo "Virtualenv: $VENV_DIR"

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  python3-pip \
  tesseract-ocr \
  tesseract-ocr-ita \
  tesseract-ocr-eng \
  libtesseract-dev \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  build-essential \
  git \
  curl \
  wget

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama already installed."
fi

if command -v systemctl >/dev/null 2>&1; then
  if ! systemctl is-active --quiet ollama 2>/dev/null; then
    echo "Starting Ollama service..."
    sudo systemctl enable ollama || true
    sudo systemctl start ollama || true
    sleep 3
  fi
fi

echo "Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
  python3.12 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python dependencies..."
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "Creating runtime directories..."
mkdir -p "$PROJECT_DIR/memory"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/chroma/financial"
mkdir -p "$PROJECT_DIR/chroma/documents"
mkdir -p "$PROJECT_DIR/chroma/drawings"
mkdir -p "$PROJECT_DIR/markdown_cache"
mkdir -p "$PROJECT_DIR/data/financial"
mkdir -p "$PROJECT_DIR/data/documents"
mkdir -p "$PROJECT_DIR/data/drawings"

deactivate

cat <<NEXT_STEPS

Setup completed.

Next steps:

1. Pull development models:

   ollama pull qwen2.5:7b
   ollama pull qwen3:0.6b

2. Start Open WebUI:

   docker run -d -p 3000:3000 \\
     --add-host=host.docker.internal:host-gateway \\
     -v open-webui:/app/backend/data \\
     --name open-webui \\
     ghcr.io/open-webui/open-webui:main

3. Start the agent server:

   source "$VENV_DIR/bin/activate"
   cd "$PROJECT_DIR/scripts"
   python server.py

4. Start the watcher in another terminal:

   source "$VENV_DIR/bin/activate"
   cd "$PROJECT_DIR/scripts"
   python watcher.py

5. In Open WebUI, configure this OpenAI-compatible endpoint:

   http://127.0.0.1:8000/v1

NEXT_STEPS