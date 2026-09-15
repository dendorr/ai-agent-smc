# AI Agent SMC — Claude Code Context

Read this entirely before touching any file.
Last updated: June 2026.

---

## Hard rules — never violate
- Never run `git push`, `git commit`, or any GitHub/remote git operation. Human pushes manually.
- Always provide complete files. No diffs, no partial snippets, no "replace lines X-Y".
- One change at a time. Implement, stop, wait for confirmation.
- Ask to see the current file before rewriting anything that already exists.
- No hardcoded domain logic. No company names, product names, or data in code.
- All configuration through environment variables via `config/config.py`.
- When a task is done: state what file was created/modified, where to copy it, what to run. Then stop.

## Code language
- All code in English: variables, functions, comments, docstrings, logs.
- LLM system prompts and user-facing text stay in Italian.
- No AI-disclaimer patterns in comments. No decorative ASCII art. No emoji.

---

## Primary mission
Build a quoting agent (agente preventivista) that:
1. Reads historical quotes and invoices from the knowledge graph
2. Reads the product catalog and pricing data
3. Reads client data (existing clients and their history)
4. Generates a pre-compiled .docx quote draft, ready for human review
5. Surfaces inside Open WebUI as agent-preventivista

This is the only feature that must work by summer 2026.
Everything else is secondary to this.

---

## Project overview
Repository: github.com/dendorr/ai-agent-smc (public, Claude Code never pushes).
Fully local, air-gapped multi-agent AI for a manufacturing SME (office partition walls, modular acoustic cabins). Zero cloud dependencies. All inference via Ollama (dev) or SGLang (production). Users interact through Open WebUI on port 3000.

Dev machine: RTX 4050 6GB, 16GB RAM, WSL2 Ubuntu 24.04, username ferra.
Production: office server, RTX 4090 24GB.
Data scope: ~1TB of relevant business documents (quotes, invoices, product catalog, client data, technical specs). Not 10TB — only the business-critical subset.

---

## Stack
- Inference (dev): Ollama, port 11434
- Inference (prod target): SGLang (same OpenAI-compatible API, zero code changes)
- Server: FastAPI async, port 8000, SSE streaming
- Interface: Open WebUI (Docker), port 3000
- Vector DB: ChromaDB + nomic-embed-text 768-dim
- Models: qwen2.5:7b (main), qwen3:0.6b (routing/fast)
- Output format: .docx via python-docx

---

## Current file structure
ai-agent/
├── config/config.py
├── scripts/
│   ├── server.py
│   ├── watcher.py
│   ├── financial_agent.py
│   ├── drawings_agent.py          # 1416 lines monolith, refactor later
│   ├── documents_agent.py
│   ├── semantic_analyzer.py
│   ├── llm_client.py              # AsyncOpenAI singleton — all LLM calls go here
│   ├── multi_step_retrieval.py
│   ├── convert_dwg.py
│   └── documents/
│       ├── init.py
│       └── ocr.py
├── requirements.txt
├── setup.sh
└── .env.example

---

## Known bugs — fix before adding features
1. scripts/documents/chunking.py does not exist. Fixed 600-word splits are in documents_agent.py directly. Semantic chunking must be created here first.
2. .env.example has OCR_ENABLED=true with OCR_MODEL=glm-ocr which does not exist in Ollama. Fix: OCR_ENABLED=false.
3. Zero tests. Every new module needs at minimum a smoke test.

---

## Python conventions
Always async. ChromaDB always with explicit OllamaEmbeddingFunction — omitting this causes silent 384 vs 768 dimension mismatch, the most dangerous bug in the codebase:

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
ef = OllamaEmbeddingFunction(url=LLM_BASE_URL.replace("/v1", ""), model_name=EMBED_MODEL)
collection = client.get_or_create_collection(name, embedding_function=ef)

ChromaDB ops are sync — wrap in executor:
result = await asyncio.get_event_loop().run_in_executor(None, collection.query, ...)

sys.path at the top of every script:
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LLM calls only via llm_client.py:
from llm_client import chat_complete, chat_complete_json, chat_complete_stream

---

## Roadmap — in this exact order
### Step 0 — Technical debt (no exceptions, do this first)
- scripts/documents/chunking.py — semantic chunking on paragraph/sentence boundaries
- .env.example — OCR_ENABLED=false as default
- SQL validator in financial_agent.py — SELECT only, block DDL/DML
- Smoke tests for chunking and filename filter

### Step 1 — Knowledge graph (minimal, quotes and finance only)
New module: scripts/knowledge/graph_builder.py
- Extracts entities from financial documents using qwen3:0.6b
- Writes to memory/knowledge_graph.sqlite
- Schema:
  nodes(id, path, title, doc_type, client, date, total_value, summary)
  edges(from_id, to_id, relation_type, weight)
  entities(id, node_id, entity_type, value, confidence)
- entity_type: client_name, project_code, product_code, unit_price, quantity, discount, payment_terms, contact_person

graph_navigator.py:
- Pure SQL traversal, no LLM calls
- Input: client name or project code
- Output: list of relevant document paths + extracted structured data

### Step 2 — Quoting agent (primary deliverable)
New file: scripts/quoting_agent.py
The agent receives natural language: "Prepara un preventivo per [cliente] per [prodotto]"

Pipeline:
1. Fast model extracts: client name, product requested, quantity, special requirements
2. Graph navigator finds: past quotes for that client, pricing for similar products
3. Main model assembles: context for the document (descriptions, notes, scope)
4. Deterministic Python calculates: totals, discounts, VAT — LLM never does math
5. python-docx generates: pre-compiled .docx with all fields populated
6. User reviews the draft and sends

Rules:
- LLM never calculates prices. All arithmetic is Python.
- Every generated quote marks which fields were inferred vs confirmed from data.
- Exposes as agent-preventivista via OpenAI-compatible API.

Output .docx structure:
- Company header (from config, not hardcoded)
- Client: name, address, contact, VAT number
- Quote number + date (auto-generated)
- Line items table: description | quantity | unit price | discount | total
- Subtotal, VAT, grand total
- Payment terms
- Validity period
- Notes (LLM-generated scope summary)
- Signature line

### Step 3 — Orchestrator (post-summer)
scripts/orchestrator_agent.py — routes queries to the right agent.
Only after Step 2 is stable and tested on real company data.

### Step 4 — Dashboard (post-summer)
Custom web UI. Only after Step 2 is stable.

---

## Models
No fine-tuning required. Structured extraction + calculation + document generation does not need a specialized model — it needs good RAG context.
- qwen2.5:7b: current main, handles Italian business docs well
- qwen3:14b: better for complex multi-product quotes, use on RTX 4090
- qwen3:0.6b: routing and entity extraction only

For JSON extraction always use chat_complete_json with a clear schema in the prompt.

---

## Environment variables
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama-no-key
LLM_MODEL_MAIN=qwen2.5:7b
LLM_MODEL_FAST=qwen3:0.6b
EMBED_MODEL=nomic-embed-text
AGENT_PORT=8000
CHUNK_SIZE=600
CHUNK_OVERLAP=60
OCR_ENABLED=false
OCR_MODEL=qwen3-vl
MULTI_STEP_ENABLED=true
MULTI_STEP_MAX_ROUNDS=1
MULTI_STEP_MIN_CONTEXT_LEN=100

---

## Start commands (WSL2)
source ~/ai-env/bin/activate
sudo systemctl restart ollama
docker start open-webui
cd ~/ai-agent/scripts
nohup env OCR_ENABLED=false python server.py > ~/ai-agent/logs/server.log 2>&1 &
nohup python watcher.py > ~/ai-agent/logs/watcher.log 2>&1 &

Re-index:
echo '{}' > ~/ai-agent/memory/watcher_registry.json
rm -rf ~/ai-agent/chroma/documents ~/ai-agent/chroma/financial ~/ai-agent/chroma/drawings
## Regole commit git
- Non aggiungere righe "Co-Authored-By" o "Claude-Session" ai messaggi di commit
- Messaggi di commit puliti, solo descrizione della modifica
