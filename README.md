# AI Agent SMC

AI Agent SMC is an experimental local multi-agent assistant designed for small and medium manufacturing companies.

The system runs on-premise and provides a ChatGPT-like interface for private document analysis, structured spreadsheet analysis, and technical drawing inspection. It is built around Open WebUI, a FastAPI OpenAI-compatible server, local LLM inference, ChromaDB, and SQLite.

This repository is a work in progress. It is a technical MVP intended to explore local AI workflows for companies that need to keep their data private.

## Goals

AI Agent SMC is designed with the following goals:

- Run locally without mandatory cloud APIs.
- Keep company data on-premise.
- Provide separate specialized agents for different business domains.
- Support private document analysis with retrieval augmented generation.
- Support structured analysis of Excel and CSV files.
- Support technical and CAD-related file inspection.
- Expose all agents through an OpenAI-compatible API.
- Integrate with Open WebUI as the user-facing chat interface.
- Keep the architecture portable across Ollama, vLLM, and SGLang.

## Architecture

The project follows a simple local architecture:

```text
Open WebUI
    |
    | OpenAI-compatible API
    v
FastAPI agent server
    |
    +-- Documents agent
    +-- Financial agent
    +-- Drawings agent
    |
    +-- ChromaDB vector indexes
    +-- SQLite structured databases
    +-- Local LLM backend
```

Typical development setup:

```text
Open WebUI -> FastAPI -> Ollama
```

Possible production setup:

```text
Open WebUI -> FastAPI -> vLLM or SGLang
```

The FastAPI server exposes OpenAI-compatible endpoints so Open WebUI can treat each agent as a selectable model.

## Agents

### Documents Agent

The documents agent handles general business documents.

Supported formats include:

- PDF
- DOCX
- PPTX
- Markdown
- TXT

Current and planned capabilities include:

- Native text extraction.
- OCR fallback for scanned documents.
- Markdown conversion cache.
- Metadata-aware retrieval.
- Filename-aware filtering.
- Multi-step retrieval for complex questions.

### Financial Agent

The financial agent handles spreadsheets, CSV files, and structured business data.

Supported formats include:

- XLSX
- XLS
- CSV
- TXT
- Financial PDFs

Main capabilities include:

- Spreadsheet parsing.
- CSV parsing.
- SQLite-based structured querying.
- Retrieval over unstructured financial text.
- Metadata extraction from tables and worksheets.

The long-term goal is to combine semantic retrieval with strict, safe SQL querying for precise tabular questions.

### Drawings Agent

The drawings agent handles technical and CAD-related files.

Supported and planned formats include:

- DXF
- DWG through conversion where available
- STEP / STP
- IFC
- STL
- SVG
- Technical PDFs

Main capabilities include:

- Technical file parsing.
- Metadata extraction from drawings.
- Filename-aware retrieval.
- Searchable descriptions of CAD and technical documents.

The long-term goal is to provide a local assistant for engineers, designers, and technical teams working with private manufacturing files.

## Repository Structure

```text
ai-agent-smc/
├── config/
│   └── config.py
├── scripts/
│   ├── server.py
│   ├── watcher.py
│   ├── financial_agent.py
│   ├── drawings_agent.py
│   ├── documents_agent.py
│   ├── semantic_analyzer.py
│   ├── llm_client.py
│   ├── multi_step_retrieval.py
│   ├── nielsen_db_builder.py
│   └── convert_dwg.py
├── data/
│   ├── financial/
│   ├── drawings/
│   └── documents/
├── chroma/
│   ├── financial/
│   ├── drawings/
│   └── documents/
├── memory/
├── logs/
├── markdown_cache/
├── requirements.txt
├── setup.sh
├── .env.example
└── README.md
```

The following folders are runtime folders and are ignored by Git:

```text
data/
memory/
markdown_cache/
chroma/
logs/
```

These folders may contain private files, generated indexes, logs, or local runtime state.

## Requirements

Recommended development environment:

- Linux or WSL2 with Ubuntu
- Python 3.12
- Docker
- Ollama
- Open WebUI

Recommended production environment:

- Linux server
- NVIDIA GPU
- Local inference backend such as vLLM or SGLang
- Open WebUI
- FastAPI agent server

## Quick Start

Clone the repository:

```bash
git clone https://github.com/dendorr/ai-agent-smc.git
cd ai-agent-smc
```

Run the setup script:

```bash
bash setup.sh
```

Copy the example environment file:

```bash
cp .env.example .env
```

Pull the default development models with Ollama:

```bash
ollama pull qwen2.5:7b
ollama pull qwen3:0.6b
```

Start the agent server:

```bash
source ~/ai-env/bin/activate
cd scripts
python server.py
```

Start the file watcher in another terminal:

```bash
source ~/ai-env/bin/activate
cd scripts
python watcher.py
```

Start Open WebUI:

```bash
docker run -d -p 3000:3000 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Configure Open WebUI to use the local OpenAI-compatible endpoint:

```text
http://127.0.0.1:8000/v1
```

The available agent models are:

```text
agent-documents
agent-financial
agent-drawings
```

## Configuration

Configuration is handled through environment variables.

Start from the example file:

```bash
cp .env.example .env
```

Main variables:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama-no-key

LLM_MODEL_MAIN=qwen2.5:7b
LLM_MODEL_FAST=qwen3:0.6b
EMBED_MODEL=nomic-embed-text

AGENT_PORT=8000

CHUNK_SIZE=600
CHUNK_OVERLAP=60

OCR_ENABLED=true
OCR_MODEL=glm-ocr

MULTI_STEP_ENABLED=true
MULTI_STEP_MAX_ROUNDS=1
MULTI_STEP_MIN_CONTEXT_LEN=100
```

For production deployments, override these values through `.env`, systemd, Docker, or shell exports.

## OpenAI-Compatible API

The FastAPI server exposes OpenAI-compatible endpoints.

### Health Check

```http
GET /health
```

### Models

```http
GET /v1/models
```

Expected model IDs:

```text
agent-documents
agent-financial
agent-drawings
```

### Chat Completions

```http
POST /v1/chat/completions
```

Example request:

```json
{
  "model": "agent-documents",
  "messages": [
    {
      "role": "user",
      "content": "Summarize the uploaded company presentation."
    }
  ],
  "stream": true
}
```

## Data Privacy

This project is designed for local and on-premise usage.

The repository should not contain private company documents, customer data, financial data, CAD files, generated vector indexes, logs, or local environment files.

Do not commit:

- Private documents.
- Customer data.
- Financial data.
- CAD files from real projects.
- Generated ChromaDB indexes.
- Runtime memory files.
- Logs containing prompts or document excerpts.
- Local `.env` files.

The `.env.example` file is safe to commit because it contains only example values.

## Current Limitations

This project is under active development.

Known limitations:

- The codebase is being refactored for readability and maintainability.
- Some retrieval modules are experimental.
- OCR and vision-based parsing require further hardening.
- SQL generation must be protected by a stricter validator before production use.
- CAD parsing coverage depends on the file format and available local converters.
- Production deployment needs stronger security defaults.
- Automated tests and CI are still missing.

## Roadmap

Near-term improvements:

- Clean and format all Python and shell files.
- Add a reproducible Docker Compose setup.
- Add a Makefile for common commands.
- Add linting and formatting configuration.
- Add basic tests.
- Add GitHub Actions for CI.
- Add configurable CORS.
- Add optional API key authentication.
- Add a strict SQL validator for LLM-generated queries.
- Add hybrid retrieval with vector search and lexical search.
- Improve citation and source reporting in answers.
- Improve CAD metadata extraction.
- Improve OCR parser modularity.

Production-oriented improvements:

- Benchmark Ollama, vLLM, and SGLang on target hardware.
- Evaluate larger local models.
- Add safer logging.
- Add deployment documentation for Linux servers.
- Add backup and restore strategy for indexes and runtime data.
- Add monitoring for latency, errors, and model performance.

## Development Notes

Code should use English for:

- File names.
- Function names.
- Variable names.
- Comments.
- Docstrings.
- Internal logs.

User-facing prompts and responses can be localized depending on the deployment context.

## License and Usage

This project is currently source-available for portfolio and demonstration purposes.

No open-source license has been selected yet.

Unless a license file is added later, all rights are reserved and the code is not licensed for commercial reuse, redistribution, or derivative works.