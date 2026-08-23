# Insurance Claims Support AI Copilot

An internal copilot for insurance claims adjusters. It reads a new claim, retrieves
three distinct kinds of context — **past approved resolutions**, **company policy
documents**, and **live operational facts** — and drafts a recommendation for a human
adjuster to edit and approve. Approved resolutions are written back into memory, so
the system gets measurably better at handling claims it has seen before.

Built with **FastAPI**, **LangGraph**, **LangMem**, **ChromaDB**, **OpenAI**, **SQLite**,
and **Streamlit**.

> **This is a copilot, not an autopilot.** The system never sets a claim status, quotes
> a settlement figure, or decides liability on its own. Every output is a draft that a
> licensed human reviews, and only human-approved text ever enters long-term memory.

---

## Table of Contents

- [The problem](#the-problem)
- [How it works](#how-it-works)
- [The three context sources](#the-three-context-sources)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Deep dive: how each layer works](#deep-dive-how-each-layer-works)
- [Design decisions](#design-decisions)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## The problem

A claims adjuster handling a new First Notice of Loss needs four things before they can
write anything useful:

1. The claim's own details
2. What the company rulebook says about this class of claim
3. How the team resolved similar claims previously
4. Live facts — the policyholder's plan tier, how many claims they already have open

Points 2 and 3 are the expensive ones. Policy knowledge sits in documents nobody
re-reads, and institutional precedent lives in the heads of senior adjusters. New
adjusters either ask someone or guess.

This system assembles all four automatically and drafts a recommendation from them,
with a full record of what it looked at.

---

## How it works

```
Adjuster registers a claim (Streamlit)
        │
        ▼
POST /tickets  ─────────────────►  FastAPI
                                      │
                                      ├─► SQLite: upsert customer, insert ticket
                                      │
                                      └─► BackgroundTask: generate_draft()
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        ▼                                       ▼                                       ▼
 1. MEMORY RETRIEVAL                   2. KNOWLEDGE RETRIEVAL                  3. TOOL CALLING
 LangMem + InMemoryStore               ChromaDB, top-k chunks                  plan tier lookup,
 "how did we resolve                   "what does the rulebook say?"           open-claim load
  claims like this before?"
        │                                       │                                       │
        └───────────────────────────────────────┼───────────────────────────────────────┘
                                                ▼
                              4. PROMPT COMPOSITION  (system rules + claim + context)
                                                ▼
                              5. OpenAI  →  four-section draft recommendation
                                                ▼
                              6. Persist draft + context_used to SQLite
                                                │
        ┌───────────────────────────────────────┘
        ▼
Adjuster reviews the draft and the full context trail → edits → APPROVE
        ▼
POST /drafts/{id}/accept  →  approved text written into LangMem
        ▼
Retrieved as precedent by the next similar claim
```

The last two steps are the point of the project. Claim #1 gets a generic draft. Claim
#20 gets a draft shaped by nineteen decisions a human already signed off on — with no
retraining, no fine-tuning, and no model changes.

---

## The three context sources

The central design idea. These look interchangeable and are not:

| Source | Answers | Storage | Update cadence | Written by |
|---|---|---|---|---|
| **Memory** (LangMem) | *"What did we decide last time?"* | LangGraph `InMemoryStore` | Every approval | Humans, via the approve action |
| **Knowledge** (RAG) | *"What does the rulebook say?"* | ChromaDB, on disk | Manual re-ingest | Authors of the policy docs |
| **Tools** | *"What is true right now?"* | SQLite, live query | Real time | The database |

A rulebook cannot tell you today's open-claim count. A SQL query cannot tell you what a
policy clause means. A vector index cannot tell you what your team decided last Tuesday.
Collapsing these into one retriever — the common shortcut — loses all three distinctions
and produces drafts that are confidently wrong about live facts.

---

## Architecture

```
┌──────────────────────┐
│  Streamlit dashboard │  register claim · review · edit · approve · probe memory
└──────────┬───────────┘
           │ HTTP
┌──────────▼───────────┐
│    FastAPI backend   │  routers → services → repositories
└──────────┬───────────┘
           │
    ┌──────┼──────────────────┬──────────────────┬─────────────────┐
    ▼      ▼                  ▼                  ▼                 ▼
┌────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────┐
│ SQLite │ │  ChromaDB    │ │   LangMem    │ │  Support   │ │   OpenAI   │
│        │ │  (RAG)       │ │  InMemory-   │ │  Tools     │ │ chat +     │
│ custo- │ │              │ │  Store       │ │            │ │ embeddings │
│ mers   │ │ policy docs  │ │ approved     │ │ plan tier  │ │            │
│ tickets│ │ chunked +    │ │ resolutions, │ │ open-claim │ │ gpt-4.1-   │
│ drafts │ │ embedded     │ │ 2 scopes     │ │ load       │ │ mini       │
└────────┘ └──────────────┘ └──────────────┘ └────────────┘ └────────────┘
```

### Layering

```
routers/        HTTP in, HTTP out. Validation via Pydantic. No logic.
services/       Orchestration and business rules. Decides the order of operations.
repositories/   SQL only. One file per table. No decisions.
integrations/   Everything external: OpenAI, ChromaDB, LangMem, tools.
```

A router never writes SQL. A repository never calls an LLM. When a draft comes out
wrong there is exactly one file to open.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, automatic OpenAPI docs, Pydantic validation for free |
| Agent runtime | LangGraph (`create_agent`) | Handles the model ⇄ tool loop and exposes the full message trace |
| Long-term memory | LangMem + `InMemoryStore` | Namespaced, semantically searchable memory with a managed write API |
| Vector store | ChromaDB (persistent) | Zero-infrastructure, on-disk, cosine similarity |
| LLM + embeddings | OpenAI `gpt-4.1-mini`, `text-embedding-3-small` | One provider, one key, one bill for both chat and embeddings |
| Database | SQLite | Single file, no server, correct choice at this scale |
| UI | Streamlit | Fast to build, and the transparency panel is the actual product |
| Tooling | uv | Fast dependency resolution and lockfile |

---

## Quickstart

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key with billing enabled

### Setup

```bash
git clone https://github.com/<your-username>/insurance-claims-copilot.git
cd insurance-claims-copilot

uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv sync
```

### Configure

```bash
cp .env.example .env
```

Then edit `.env` and add your key:

```bash
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536
ENABLE_SEMANTIC=true
API_BASE_URL=http://localhost:8000
```

### Run

Two terminals.

```bash
# Terminal 1 — API
uv run uvicorn customer_support_agent.main:app --reload --port 8000

# Terminal 2 — dashboard
uv run streamlit run app.py
```

- Dashboard: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

### First run

1. In the dashboard sidebar, click **Ingest knowledge base**. This chunks and embeds
   the policy documents in `knowledge_base/`. Run it once, and again whenever you edit
   those files.
2. Go to **Register claim** and file one.
3. Go to **Review drafts**, select it, and click **Generate**.
4. Read the draft — then open the **Context used by the copilot** panel below it.
5. Edit if needed, tick **Save approved resolution to memory**, and click **Approve**.
6. File a second, similar claim and watch the memory hits change the draft.

Step 6 is the demo. Everything before it is setup.

---

## Configuration

All settings live in `customer_support_agent/core/settings.py` and are overridable via
`.env`.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Used for both chat and embeddings |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Chat model for draft generation |
| `LLM_TEMPERATURE` | `0.2` | Low, because drafts should be consistent |
| `LLM_TIMEOUT` | `60` | Seconds |
| `LLM_MAX_RETRIES` | `3` | Backoff on rate limits |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Used by RAG **and** memory |
| `EMBEDDING_DIMS` | `1536` | Pinned so Chroma and the memory index agree |
| `EMBEDDING_BATCH_SIZE` | `64` | Chunks per embedding request |
| `ENABLE_SEMANTIC` | `true` | Set `false` to run fully degraded, with no embedding calls |
| `CHUNK_SIZE` | `900` | Characters per knowledge chunk |
| `CHUNK_OVERLAP` | `150` | Overlap so rules aren't lost at boundaries |
| `RAG_TOP_K` | `4` | Knowledge chunks per draft |
| `MEMORY_TOP_K` | `4` | Memories per draft |

> **Changing `EMBEDDING_MODEL` or `EMBEDDING_DIMS` is a breaking change.** Vectors of
> different widths cannot coexist in one Chroma collection or one store index. Delete
> `data/chroma/`, re-ingest, and restart the API.

---

## API reference

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, configured model, whether semantic mode is active |

### Tickets (claims)

| Method | Path | Description |
|---|---|---|
| `POST` | `/tickets` | Register a claim. Upserts the customer and triggers background draft generation |
| `GET` | `/tickets` | List claims, newest first |
| `GET` | `/tickets/{id}` | Fetch one claim |

### Knowledge base

| Method | Path | Description |
|---|---|---|
| `POST` | `/knowledge/ingest` | Chunk, embed, and upsert every `.md` file in `knowledge_base/`. Idempotent |
| `GET` | `/knowledge/stats` | Indexed chunk count and semantic status |
| `GET` | `/knowledge/search?q=&top_k=` | Raw retrieval, for debugging |

### Drafts

| Method | Path | Description |
|---|---|---|
| `POST` | `/drafts/generate/{ticket_id}` | Generate a draft synchronously |
| `GET` | `/drafts/ticket/{ticket_id}` | Latest draft for a claim, with `context_used` |
| `PATCH` | `/drafts/{id}` | Save an edit without approving |
| `POST` | `/drafts/{id}/accept` | Approve. Optionally writes the resolution to memory |
| `POST` | `/drafts/{id}/discard` | Discard. Nothing is written to memory |

### Memory

| Method | Path | Description |
|---|---|---|
| `GET` | `/memory/search?q=&email=&company=` | Probe both memory scopes directly |

**Example — registering a claim**

```bash
curl -X POST http://localhost:8000/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "name": "Ravi Kumar",
      "email": "ravi@acme.com",
      "company": "Acme Ltd",
      "plan_tier": "premium"
    },
    "subject": "Windshield cracked by road debris",
    "body": "A stone hit my windshield on the highway. The car is drivable, no injuries.",
    "claim_type": "auto",
    "priority": "medium"
  }'
```

**Example — approving with memory write-back**

```bash
curl -X POST http://localhost:8000/drafts/1/accept \
  -H "Content-Type: application/json" \
  -d '{"content": "<final edited text>", "save_to_memory": true}'
```

---

## Project structure

```
insurance-claims-copilot/
├── customer_support_agent/
│   ├── core/
│   │   └── settings.py                  # Pydantic settings, single source of config
│   ├── db/
│   │   └── database.py                  # SQLite connection + schema
│   ├── models/
│   │   └── schemas.py                   # Pydantic request/response models
│   ├── repositories/
│   │   ├── customer_repo.py             # SQL only
│   │   ├── ticket_repo.py
│   │   └── draft_repo.py
│   ├── integrations/
│   │   ├── llm/openai_client.py         # ChatOpenAI + OpenAIEmbeddings singletons
│   │   ├── rag/chroma_kb.py             # Chunk, embed, upsert, search
│   │   ├── memory/langmem_store.py      # Namespaces, write, scoped search, fallbacks
│   │   └── tools/support_tools.py       # @tool functions the agent can call
│   ├── services/
│   │   ├── copilot_service.py           # Context gathering, prompt, agent loop
│   │   └── draft_service.py             # Draft lifecycle + memory write-back
│   ├── api/
│   │   ├── app_factory.py               # App construction, CORS, lifespan
│   │   └── routers/
│   │       ├── health.py
│   │       ├── tickets.py
│   │       ├── drafts.py
│   │       ├── knowledge.py
│   │       └── memory.py
│   └── main.py                          # ASGI entry point
├── knowledge_base/                      # Policy documents, markdown
│   ├── insurance-auto-claims-fnol-intake-checklist.md
│   ├── insurance-auto-coverage-and-deductible-guidelines.md
│   ├── insurance-auto-required-documents-by-claim-type.md
│   ├── insurance-claims-settlement-sla-and-communication.md
│   └── insurance-claims-fraud-risk-indicators.md
├── app.py                               # Streamlit dashboard
├── data/                                # SQLite file + Chroma index (gitignored)
├── .env.example
└── pyproject.toml
```

---

## Deep dive: how each layer works

### Knowledge retrieval (RAG)

Ingestion is four stages:

1. **Read** — every `.md` file in `knowledge_base/`
2. **Chunk** — `RecursiveCharacterTextSplitter`, 900 characters with 150 overlap, split
   preferentially at `##` headings so a rule is never cut in half
3. **Embed** — batched calls to `text-embedding-3-small`, 1536 dimensions per chunk
4. **Store** — `upsert` into ChromaDB with deterministic IDs (`filename::index`)

Deterministic IDs make re-ingestion idempotent: editing a knowledge file and re-running
overwrites the affected chunks instead of duplicating them.

At query time the claim's subject and body are embedded with the *same* model and the
nearest chunks are returned with their source filename, which the system prompt requires
the model to cite.

### Memory (LangMem)

Every approved resolution is written to **two namespaces**:

```python
("claims", "customer", "ravi-acme-com")      # this policyholder's history
("claims", "company",  "company-acme-ltd")   # the whole organisation's history
```

Customer scope answers *"what happened with this person before."* Company scope answers
*"how does our team handle this."* A first-time claimant at a known company still
benefits from their colleagues' precedents.

Emails are slugified for namespace use because LangGraph rejects periods in namespace
labels, while the un-slugified form remains the SQLite key. Both derive from the same
normalised lowercase email, so one person can never split into two memory scopes.

Retrieval searches both scopes, deduplicates on a text fingerprint, sorts by score, and
truncates to `MEMORY_TOP_K`.

### Tool calling

Two tools are exposed to the agent:

- `lookup_customer_plan(email)` — plan tier and the SLA that applies
- `lookup_open_ticket_load(email)` — open-claim count, with a duplicate-filing warning
  above three

The model decides whether to call them by reading their **docstrings**, which are sent
to the API as tool descriptions. Both return prose rather than JSON, because the output
goes directly into the model's context and should read like something a colleague said.

### Draft generation

`copilot_service.generate_recommendation()` runs five steps: gather context, format it,
build the prompt, run the agent loop, and fall back if the agent produces nothing.

The system prompt encodes the guardrails:

- Ground every claim in provided context; write `"Needs verification: ..."` rather than
  guessing
- Never quote or imply a settlement amount
- Never state a final liability or coverage decision
- Cite the knowledge source filename in brackets
- Output four fixed sections, under 300 words

Every memory hit, knowledge chunk, and tool call is captured into `context_used` and
persisted alongside the draft. That record is what the dashboard's transparency panel
renders.

### Human-in-the-loop

The adjuster can edit freely, save without approving, discard, or approve. Approval
does two things: marks the ticket resolved, and — **only if the human leaves the
checkbox ticked** — writes the final text into memory.

A discarded draft never enters memory. That single rule is what keeps retrieval quality
from degrading over time.

---

## Design decisions

**Three retrieval channels, not one.** Documents, precedent, and live facts have
different freshness requirements and different failure modes. Merging them into a single
vector index would make the system confidently stale about facts that change hourly.

**Failures degrade, they don't cascade.** Memory retrieval, knowledge retrieval, and the
agent run are each wrapped independently. Any of them can fail and a draft is still
produced, with the failure recorded in `context_used.errors`. The layered fallbacks:

| Condition | Behaviour |
|---|---|
| Semantic on, memories match | Ranked semantic results |
| Semantic on, no match | Recent-memory listing |
| `ENABLE_SEMANTIC=false` or no key | Plain store, recent listing only |
| Embeddings endpoint errors | Logged, plain store, draft continues |
| Agent returns empty text | Plain LLM call without tools |
| LLM entirely unavailable | Static placeholder draft, error recorded |

**Ingestion is explicit, never automatic.** Policy documents are static. Re-embedding on
every boot would burn money and startup time for nothing, so ingestion is a deliberate
human action — a button and a `POST` endpoint.

**Drafts are a separate table from tickets.** One claim can have several drafts across
regeneration and discard cycles, each carrying its own context record. Merging them
would throw that history away.

**Only humans write to memory.** The agent can read memory but has no memory-write tool.
Every entry passed a human review gate.

**Background generation.** `POST /tickets` returns `201` immediately; drafting runs
after the response. The adjuster is never blocked on a 15-second LLM call.

---

## Known limitations

Stated plainly, because they were deliberate trade-offs rather than oversights.

- **Memory does not survive a restart.** `InMemoryStore` is RAM-only. Swapping to a
  Postgres-backed store is a single-line change in `get_store()`; the in-memory version
  was chosen to keep the demo dependency-free. ChromaDB and SQLite *do* persist.
- **No authentication.** The API is unauthenticated and CORS is fully open. Do not
  expose it publicly without adding an API key or JWT layer.
- **Single-node only.** SQLite and an in-process memory store rule out horizontal
  scaling as-is.
- **The knowledge base is invented policy.** Plausible and internally consistent, but
  not real insurance regulation. Replace it before any real use.
- **No automated evaluation.** Draft quality is judged by reading. Groundedness scoring
  and edit-distance tracking would make quality measurable.
- **No reranking.** Top-k chunks go straight into the prompt without a second-stage
  relevance pass.

---


## 📄 License

This project is for educational and demonstration purposes. Feel free to fork and extend.
This project is for educational purposes. Feel free to use and adapt it for your own learning.



## ⭐ If you found this helpful

Give this repository a star ⭐ 
