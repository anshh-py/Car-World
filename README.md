# 🚗 Car World AI

### AI-Powered Automotive Research, Discovery & Recommendation Platform

<p align="center">
  <a href="https://car-world-1.onrender.com/"><strong>Live Demo</strong></a> •
  <a href="https://github.com/anshh-py/Car-World"><strong>GitHub Repository</strong></a> •
  <a href="https://huggingface.co/spaces/ansh9057/carworld-qwen3-inference"><strong>Qwen3 Inference</strong></a> •
  <a href="https://huggingface.co/ansh9057/carworld-qwen3-4b-lora"><strong>Model Adapter</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge" alt="Groq"/>
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white" alt="Supabase"/>
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=black" alt="Render"/>
</p>

---

## Overview

**Car World AI** is an AI-powered automotive research and discovery platform that helps users explore vehicles, compare specifications, and ask automotive questions through a conversational interface.

Instead of depending entirely on static filters and manual specification searches, users can interact with an AI assistant using natural language.

The platform combines:

- LLM-powered conversational AI.
- Tool-based agent execution.
- Structured automotive information retrieval.
- External web search integration.
- Fine-tuned Qwen3-4B model integration.
- Multi-provider inference with automatic fallback and recovery.
- Backend security and performance instrumentation.

A central engineering objective is **LLM provider reliability**.

Groq serves as the primary inference provider, using `openai/gpt-oss-120b`. When the primary provider encounters HTTP 429 rate limits, the backend can route requests to a separately hosted fine-tuned Qwen3-4B LoRA model on Hugging Face Spaces.

The provider router also implements cooldown-based recovery, allowing the application to resume using its primary provider.

## Live Application

| Component | Link |
|---|---|
| Frontend | [Car World AI](https://car-world-1.onrender.com/) |
| Backend | [FastAPI Backend](https://car-world-agoj.onrender.com) |
| Health Check | [API Health](https://car-world-agoj.onrender.com/health) |
| Source Code | [GitHub Repository](https://github.com/anshh-py/Car-World) |
| Qwen3 Inference | [Hugging Face Space](https://huggingface.co/spaces/ansh9057/carworld-qwen3-inference) |
| Fine-tuned Adapter | [Qwen3-4B LoRA](https://huggingface.co/ansh9057/carworld-qwen3-4b-lora) |

---

## Key Features

### 1. AI-Powered Automotive Assistant

- Natural-language automotive conversations.
- Context-aware conversation history.
- Automotive discovery and recommendation assistance.
- Streaming conversational responses.
- Tool-based execution for structured information retrieval.
- Integrated web search for additional information.

### 2. Multi-Provider LLM Architecture

- Groq-hosted `openai/gpt-oss-120b` as the primary model.
- Fine-tuned Qwen3-4B LoRA model as the fallback provider.
- Remote fallback inference through Hugging Face Spaces.
- Automatic fallback on primary-provider HTTP 429 responses.
- Cooldown-based recovery to the primary provider.
- Backend-managed provider selection and orchestration.

### 3. Automotive Information Retrieval

- PostgreSQL-backed structured automotive information.
- Tool-based database queries.
- Tavily-powered external web search.
- Backend-managed retrieval and response generation.

### 4. Security Engineering

The backend implements multiple layers of application and AI-specific security controls:

- Prompt-injection and jailbreak detection.
- Boundaries between trusted instructions and untrusted content.
- Tool allowlisting and argument validation.
- Request-size protection.
- Output validation.
- Personally identifiable information (PII) redaction.
- Abuse-gating controls.
- Security evaluation utilities.

These controls follow a defense-in-depth approach rather than relying exclusively on model-level instruction following.

### 5. Performance & Observability

The application includes instrumentation for understanding inference performance and resource usage.

| Metric | Purpose |
|---|---|
| Provider latency | Measure individual inference-provider response time |
| Request latency | Track end-to-end request duration |
| Tool execution | Identify retrieval and tool bottlenecks |
| Agent initialization | Measure agent setup overhead |
| Cache lookup | Observe semantic-cache activity |
| Usage accounting | Estimate provider usage and configurable costs |

Additional utilities include:

- Prompt-token benchmarking.
- Deterministic token estimation.
- Aggregate diagnostic logging.
- Offline batch execution with bounded concurrency, throttling, and retries.

Token estimation is heuristic-based and intended for diagnostics, not billing-grade accounting.

### 6. Optional Semantic Caching

A privacy-conscious semantic cache is available for guest interactions.

- Disabled by default.
- Configurable through environment variables.
- TTL-based expiration.
- Bounded cache size.
- Configurable similarity threshold.
- Designed to avoid caching authenticated users' private conversations.

---

## System Architecture

Car World AI separates the frontend, API layer, security validation, AI orchestration, provider routing, tools, and data storage.

```mermaid
flowchart TD
    A["User"] --> B["Web Frontend"]
    B --> C["FastAPI Backend"]

    C --> D["Authentication & API Controls"]
    D --> E["Security Validation"]
    E --> F["AI Agent / Orchestration"]

    F --> G["Provider Router"]

    G --> H["Groq"]
    H --> I["GPT-OSS-120B"]

    G --> J["Fallback Provider"]
    J --> K["Hugging Face Space"]
    K --> L["Fine-tuned Qwen3-4B"]

    F --> M["Tool Execution"]
    M --> N["PostgreSQL"]
    M --> O["Tavily Web Search"]

    N --> P["Retrieved Information"]
    O --> P

    P --> F

    I --> Q["Response Generation"]
    L --> Q

    Q --> R["API Response"]
    R --> B
```

### Architecture Components

| Component | Responsibility |
|---|---|
| Frontend | Conversational automotive interface |
| FastAPI | HTTP API and application integration |
| Security Layer | Input validation and abuse controls |
| AI Agent | Model interaction and tool execution |
| Provider Router | Primary/fallback selection and recovery |
| Groq | Primary LLM inference |
| Hugging Face | Remote fallback inference |
| PostgreSQL | Structured automotive information |
| Tavily | External web search |
| Supabase | Authentication integration |

---

## AI Provider Routing

A central design decision is to avoid making the application dependent on a single inference provider.

The backend implements a primary-and-fallback strategy.

```mermaid
flowchart TD
    A["Incoming Request"] --> B["Provider Router"]

    B --> C["Groq: GPT-OSS-120B"]

    C -->|Successful Response| D["Return Primary Response"]
    C -->|HTTP 429 Rate Limit| E["Activate Fallback"]

    E --> F["Hugging Face Space"]
    F --> G["Fine-tuned Qwen3-4B"]
    G --> H["Return Fallback Response"]

    E --> I["Cooldown Period"]
    I --> J["Primary Provider Recovery"]
    J --> C
```

| Provider | Model | Role |
|---|---|---|
| Primary | `openai/gpt-oss-120b` via Groq | Main conversational inference |
| Fallback | Fine-tuned Qwen3-4B LoRA | Alternative inference during primary rate limits |
| Hosting | Hugging Face Spaces | Remote fallback inference |

Provider-specific handling remains inside the backend, keeping the frontend independent of inference-provider selection.

---

## Security Architecture

LLM applications introduce risks beyond conventional API security. User messages, retrieved content, and tool arguments can influence model behavior.

Car World AI uses layered security controls to reduce these risks.

```mermaid
flowchart TD
    A["Incoming Request"] --> B["Request Size & Abuse Controls"]
    B --> C["Input Security Checks"]
    C --> D["Prompt Injection / Jailbreak Detection"]
    D --> E["AI Agent"]

    E --> F["Untrusted Data Boundaries"]
    F --> G["Tool Allowlist"]
    G --> H["Tool Argument Validation"]

    H --> I["Database / Web Search"]
    I --> J["Response Generation"]
    J --> K["Output Validation"]
    K --> L["PII Redaction"]
    L --> M["API Response"]
```

Security mechanisms include:

- Suspicious instruction-pattern detection.
- Separation of untrusted content from trusted application instructions.
- Validation before tool execution.
- Output checks before returning responses.
- PII redaction and abuse controls.

These mechanisms provide defense in depth. They reduce risk but do not guarantee detection of every possible attack.

---

## Performance & Observability

Performance instrumentation focuses on identifying where time and model usage are spent during an AI request.

| Metric Area | Purpose |
|---|---|
| Provider latency | Measure inference-provider response time |
| Total request latency | Track end-to-end request duration |
| Tool execution | Identify retrieval bottlenecks |
| Agent initialization | Track setup overhead |
| Cache lookup | Observe optional caching activity |
| Usage accounting | Estimate provider usage and costs |

### Supporting Utilities

| Utility | Purpose |
|---|---|
| Prompt benchmark | Reproducible prompt-token comparison |
| Token estimator | Deterministic heuristic estimation |
| Batch runner | Offline execution with bounded concurrency and retries |
| Diagnostics | Aggregate performance and usage logging |
| Semantic cache | Optional guest-only response caching |

The batch runner is a standalone offline utility, not part of the live conversational request pipeline.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Primary LLM | GPT-OSS-120B via Groq |
| Fallback Model | Qwen3-4B with LoRA |
| Model Hosting | Hugging Face Spaces |
| Model Integration | Gradio Client |
| Database | PostgreSQL |
| Authentication | Supabase |
| Web Search | Tavily |
| Database Migrations | Alembic |
| Testing | Pytest |
| Deployment | Render |
| Version Control | Git, GitHub |

---

## Fine-Tuned Model Integration

Car World AI integrates a Qwen3-4B model with a LoRA adapter as its secondary inference provider.

The inference service is hosted separately on Hugging Face Spaces and accessed by the backend using the Gradio Client.

**Model resources:**

- [Qwen3 Inference Space](https://huggingface.co/spaces/ansh9057/carworld-qwen3-inference)
- [Qwen3-4B LoRA Adapter](https://huggingface.co/ansh9057/carworld-qwen3-4b-lora)

This integration demonstrates experience with:

- Fine-tuned model integration.
- LoRA adapter-based model workflows.
- Remote model inference.
- Provider abstraction.
- Fallback handling between independent inference services.

---

## Backend API

The application exposes a REST API through FastAPI.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend health check |
| `/api/v1/health` | GET | Health compatibility endpoint |
| `/api/chat` | POST | Conversational AI interaction |

### Example Request

```json
{
  "message": "Compare two cars based on their specifications",
  "history": []
}
```

### Example Python Client

```python
import requests

url = "https://car-world-agoj.onrender.com/api/chat"

payload = {
    "message": "Recommend a car for city driving",
    "history": []
}

response = requests.post(
    url,
    json=payload,
    timeout=60
)

response.raise_for_status()

print(response.json())
```

---

## Project Structure

The repository separates the backend application, frontend, tests, documentation, and supporting utilities.

```text
Car-World/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── agents/
│   │   ├── services/
│   │   ├── providers/
│   │   ├── database/
│   │   ├── core/
│   │   └── main.py
│   │
│   ├── tests/
│   ├── scripts/
│   └── requirements.txt
│
├── apps/
│   └── Frontend application assets
│
├── docs/
│   ├── OPTIMIZATION_REPORT.md
│   └── DEPLOYMENT.md
│
├── app.py
├── .env.example
├── Dockerfile
└── README.md
```

*This is a high-level structural overview, not an exhaustive listing of every repository file.*

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/anshh-py/Car-World.git
cd Car-World
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using `.env.example` as the reference.

Configure the credentials and connection details required by the services you intend to use:

- Groq API key.
- Hugging Face credentials and Space configuration.
- PostgreSQL connection.
- Supabase credentials, if using authentication.
- Tavily API key, if using web search.

Optional performance configuration includes:

- `SEMANTIC_CACHE_ENABLED`
- `SEMANTIC_CACHE_TTL_SECONDS`
- `SEMANTIC_CACHE_MAX_ENTRIES`
- `SEMANTIC_CACHE_SIMILARITY_THRESHOLD`
- `GROQ_*_PRICE_PER_1M`
- `QWEN3_*_PRICE_PER_1M`

Semantic caching is disabled by default.

Refer to `.env.example` and `docs/DEPLOYMENT.md` for the actual configuration details.

**Never commit API keys, passwords, access tokens, or other secrets to GitHub.**

### 5. Start the Backend

From the repository root:

```bash
python -m uvicorn --app-dir backend app.main:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

### 6. Start the Frontend

On Windows:

```bash
start-frontend.cmd
```

Frontend:

```text
http://127.0.0.1:5500
```

### 7. Run Tests

```bash
cd backend
python -m pytest -q
```

For optional live fallback-provider verification:

```bash
python test_live_qwen_fallback.py
```

This interactive verification requires the relevant Hugging Face credentials and a reachable inference service.

---

## Deployment

Car World AI uses independently hosted application and inference components.

| Service | Platform | Responsibility |
|---|---|---|
| Frontend | Render | User-facing application |
| Backend | Render | FastAPI application |
| Database | PostgreSQL / Supabase | Structured data storage |
| Primary Inference | Groq | Main LLM inference |
| Fallback Inference | Hugging Face Spaces | Qwen3 model inference |
| Authentication | Supabase | Authentication integration |

### Deployment Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Optimization Report](docs/OPTIMIZATION_REPORT.md)

---

## Engineering Highlights

This project demonstrates practical experience across applied AI and backend engineering:

- Building an LLM-powered automotive application.
- Designing a modular FastAPI backend.
- Implementing tool-based AI agent execution.
- Integrating structured database retrieval with conversational AI.
- Integrating a fine-tuned Qwen3 model using LoRA.
- Implementing multi-provider LLM routing.
- Handling primary-provider rate limits and fallback recovery.
- Integrating remote inference through the Gradio Client.
- Implementing prompt-injection and jailbreak security checks.
- Validating tool names and arguments.
- Adding output validation and PII redaction.
- Instrumenting provider, tool, and request latency.
- Implementing configurable usage and cost accounting.
- Building optional privacy-conscious semantic caching.
- Writing automated security and performance regression tests.
- Deploying interconnected services using cloud platforms.

---

## Known Scope & Limitations

The project has deliberate boundaries that are important when evaluating its implementation:

- Primary and fallback models are independently hosted services.
- Fallback routing is designed around primary-provider HTTP 429 rate limits; it does not guarantee uninterrupted availability for every provider failure.
- Semantic caching is optional and disabled by default.
- Token estimation is heuristic-based, not billing-grade.
- The batch runner is a standalone offline utility, not part of the live chat pipeline.
- Security checks provide defense in depth, not guaranteed protection against all attacks.
- Actual model quality and response accuracy depend on the underlying models and retrieved information.

---

## Future Development

Potential areas for further development include:

- More advanced automotive recommendation algorithms.
- Improved vehicle entity recognition and comparison.
- Expanded and regularly updated automotive datasets.
- Additional model-provider integrations.
- More extensive AI response evaluation.
- Richer vehicle analytics and discovery interfaces.
- Further optimization of inference latency and operating costs.

---

## Connect

- **GitHub:** [@anshh-py](https://github.com/anshh-py)
- **Project Repository:** [Car World AI](https://github.com/anshh-py/Car-World)

<p align="center">
  <strong>Car World AI — Bringing conversational AI to automotive discovery.</strong>
</p>
