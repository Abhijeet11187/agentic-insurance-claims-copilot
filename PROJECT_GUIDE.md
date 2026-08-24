# Project Guide — Insurance Claims Support AI Copilot

This document is a complete walkthrough of the project for someone who has never seen
the code before. Read it top to bottom once, and you should understand what the system
does, how a request travels through it, and what every file and folder is for.

---

## 1. What is this project, in plain English?

This is an **internal tool for insurance claims adjusters** — not a customer-facing
chatbot. An adjuster registers a new claim (a "First Notice of Loss"), and the system
automatically writes a **draft recommendation** for how to handle it. The adjuster then
reads the draft, edits it if needed, and clicks **Approve** or **Discard**.

The interesting part is *what the draft is based on*. Before writing anything, the
system gathers three different kinds of information:

1. **Memory** — "How did we resolve similar claims before?" (based on past decisions
   humans already approved)
2. **Knowledge** — "What does the company policy rulebook say?" (from Markdown
   documents, searched like a search engine)
3. **Tools** — "What is true right now?" (e.g., this customer's plan tier, how many
   other claims they currently have open — looked up live from the database)

It combines all three, hands them to an LLM (OpenAI's `gpt-4.1-mini`), and asks it to
write a structured draft. Crucially: **the AI never makes the final decision.** It
never states a settlement amount or a liability decision — only a human can approve a
draft, and only *approved* text is saved back into memory. That feedback loop is the
whole point: claim #20 produces a much better draft than claim #1, because it can draw
on nineteen decisions a human already signed off on.

### Why this is more than "a RAG chatbot"

A lot of AI-copilot demos just shove everything into one vector database. This project
deliberately keeps three separate retrieval systems because they answer different
questions and go stale at different speeds:

| Source | Answers | Where it lives | How fresh |
|---|---|---|---|
| Memory (LangMem) | "What did we decide last time?" | In-RAM store, written only by human approval | Grows every time a human approves a draft |
| Knowledge (RAG / ChromaDB) | "What does the rulebook say?" | On-disk vector database | Static — only changes when someone re-ingests the docs |
| Tools | "What is true right now?" | SQLite, queried live | Always current |

If you merged these into one index, you'd lose the ability to tell "policy" apart from
"precedent" apart from "live fact" — and the model would end up confidently wrong about
things that change by the hour (like how many claims someone currently has open).

---

## 2. The request flow — what happens when an adjuster uses the app

```
1. Adjuster fills out a claim form in the Streamlit dashboard (app.py)
        │
        ▼
2. Streamlit sends  POST /tickets  to the FastAPI backend
        │
        ├─► FastAPI saves the customer + the claim ("ticket") into SQLite
        │
        └─► FastAPI kicks off a BACKGROUND task: "go draft a recommendation"
                    (the API responds immediately — the adjuster isn't stuck waiting)
                    │
        ┌───────────┼────────────────────────────┐
        ▼           ▼                             ▼
  MEMORY SEARCH   KNOWLEDGE SEARCH            TOOL CALLS
  (LangMem)       (ChromaDB / RAG)            (SQLite lookups)
  "similar past   "relevant policy            "this customer's plan tier,
   resolutions"    document chunks"            their open-claim count"
        │           │                             │
        └───────────┴─────────────┬───────────────┘
                                   ▼
                    All of this gets assembled into one big
                    prompt and sent to OpenAI (gpt-4.1-mini)
                                   ▼
                    The model replies with a 4-section draft:
                    Summary / Next steps / Info to request / Risk notes
                                   ▼
                    The draft (plus a full record of exactly what
                    memory/knowledge/tools were used) is saved to SQLite
        │
        ▼
3. Adjuster opens the "Review drafts" tab, reads the draft AND the
   "Context used" panel (proof of what the AI looked at)
        │
        ▼
4. Adjuster edits the text if needed, then either:
     - clicks DISCARD  → nothing is saved to memory, claim stays as-is
     - clicks APPROVE  → claim marked resolved, and (if the checkbox is
                          ticked) the final text is written into memory
                          for the next similar claim to find
```

That last step — writing approved text back into memory — is what makes the system
improve over time without ever retraining or fine-tuning a model.

---

## 3. Top-level folder structure

```
Code/
├── app.py                       # The Streamlit dashboard (what the adjuster sees)
├── main.py                      # Leftover default file from `uv init` — not used by the app
├── check_1.py … check_5.py      # Manual test/verification scripts from development
├── pyproject.toml               # Project dependencies (managed by `uv`)
├── uv.lock                      # Exact locked dependency versions
├── .env                         # Your local secrets (API key) — NOT committed to git
├── .python-version              # Pins the Python version for `uv`
├── .gitignore
├── README.md                    # Full project documentation (very detailed)
├── road_map.md                  # Step-by-step tutorial this project was originally built from
├── knowledge_base/              # Markdown "policy documents" the RAG system indexes
├── data/                        # SQLite database file + ChromaDB vector index (generated, gitignored)
├── tests/                       # Placeholder for pytest tests
└── customer_support_agent/      # The actual backend application (FastAPI service)
```

### The files at the root, one by one

| File | What it's for |
|---|---|
| `app.py` | The Streamlit UI. This is what you run to get the web dashboard at `localhost:8501`. It talks to the FastAPI backend purely over HTTP (using `requests`) — it holds no business logic of its own. |
| `main.py` | The generic placeholder file `uv init` creates for every new project (`print("Hello from code!")`). It's not part of the running application — the real backend entry point is `customer_support_agent/main.py`. |
| `check_1.py` – `check_5.py` | Small standalone scripts written during development to manually verify each layer worked before moving to the next (settings load correctly, database read/write works, memory read/write works, tools return sensible text, and the full draft-generation pipeline produces output). They correspond to the "Checkpoint" sections in `road_map.md`. Handy as runnable examples of how to call each module directly from Python, without going through the API. |
| `pyproject.toml` | Declares the project's dependencies — FastAPI, LangChain/LangGraph, LangMem, ChromaDB, Streamlit, etc. — managed with the `uv` package manager. |
| `uv.lock` | The exact resolved versions of every dependency, so installs are reproducible. |
| `.env` | Holds your real `OPENAI_API_KEY` and other local configuration. Never committed — see `.gitignore`. |
| `README.md` | The authoritative, in-depth project documentation — architecture diagrams, API reference, design-decision rationale, known limitations. Read this alongside this guide for the deepest detail. |
| `road_map.md` | A phase-by-phase build tutorial ("Phase 0 → Phase 12") written to teach someone how to build this exact system from scratch. Useful for understanding *why* each piece was added and in what order, though a couple of early implementation details in it (it originally sketched Groq + Gemini) were later swapped for OpenAI throughout the real code — trust the actual source files and README over the roadmap's code snippets where they disagree. |
| `knowledge_base/` | The "policy rulebook" — plain Markdown files that get chunked, embedded, and indexed so the AI can search them (see §5). |
| `data/` | Generated at runtime: `app.db` (the SQLite database) and `chroma/` (the on-disk vector index). Gitignored because it's local state, not source code. |
| `tests/` | Currently just `__init__.py` — a placeholder package for future pytest tests. |

---

## 4. The backend application: `customer_support_agent/`

This is a FastAPI service, organized in **layers**. Each layer has exactly one job, and
the rule is strict: a layer only talks to the layer directly below it.

```
customer_support_agent/
├── main.py                          # ASGI entry point — what uvicorn actually runs
├── core/
│   └── settings.py                  # ALL configuration lives here, nowhere else
├── db/
│   └── database.py                  # SQLite connection + table schema
├── models/
│   └── schemas.py                   # Pydantic request/response shapes
├── repositories/                    # Layer: raw SQL, one file per table
│   ├── customer_repo.py
│   ├── ticket_repo.py
│   └── draft_repo.py
├── integrations/                    # Layer: everything that talks to an external system
│   ├── llm/openai_client.py         # OpenAI chat model + embeddings
│   ├── rag/chroma_kb.py             # ChromaDB: chunk, embed, search
│   ├── memory/langmem_store.py      # LangMem: namespaced long-term memory
│   └── tools/support_tools.py       # Functions the AI agent is allowed to call
├── services/                        # Layer: business logic / orchestration
│   ├── copilot_service.py           # THE BRAIN — gathers context, runs the agent
│   └── draft_service.py             # Draft lifecycle: generate / approve / discard
└── api/                              # Layer: HTTP in, HTTP out
    ├── app_factory.py                # Builds the FastAPI app, wires routers together
    └── routers/
        ├── health.py                 # GET /health
        ├── tickets.py                 # POST/GET /tickets — register & list claims
        ├── drafts.py                  # generate / edit / approve / discard drafts
        ├── knowledge.py               # ingest & search the policy knowledge base
        └── memory.py                  # probe the memory store directly
```

### Why layers, and why this exact order?

```
routers/        HTTP in, HTTP out. Parses/validates the request. No business logic.
services/       Decides WHAT should happen and in what order. The orchestration layer.
repositories/   SQL only. One file per database table. No decision-making.
integrations/   Talks to everything outside this codebase (OpenAI, ChromaDB, LangMem).
```

A **router** never writes raw SQL — it calls a service. A **repository** never calls an
LLM — it only reads/writes rows. This means when a draft comes out looking wrong, there
is exactly one place to look (`services/copilot_service.py`), not five files scattered
across the codebase. It also makes each piece independently testable: you can test
`chroma_kb.py`'s search function without spinning up FastAPI at all (that's literally
what `check_4.py` does).

### File-by-file explanation

#### `main.py` — the entry point
```python
from customer_support_agent.api.app_factory import create_app
app = create_app()
```
This is the object `uvicorn` runs (`uvicorn customer_support_agent.main:app`). It's
deliberately a one-liner — all the real setup work lives in `app_factory.py`.

#### `core/settings.py` — the single source of configuration
Every tunable value in the whole app — the OpenAI model name, chunk sizes, top-k
retrieval counts, file paths — is declared here as a `pydantic-settings` class, and
loaded once from `.env`. Nothing else in the codebase reads environment variables
directly; everything calls `get_settings()`. This means if you want to know every knob
the app has, you read exactly one file. Key fields:
- `openai_api_key`, `openai_model` (`gpt-4.1-mini`), `llm_temperature`
- `embedding_model` (`text-embedding-3-small`), `embedding_dims` (`1536`)
- `enable_semantic` — a kill-switch to run with zero embedding calls
- `chunk_size` / `chunk_overlap` / `rag_top_k` — RAG tuning
- `memory_top_k` — how many memories to retrieve per draft
- `db_path`, `chroma_path`, `knowledge_base_dir` — where files live on disk

#### `db/database.py` — the database connection and schema
Defines the three SQLite tables as plain SQL (`customers`, `tickets`, `drafts`), and a
`get_conn()` context manager that every repository uses to open a connection, commit on
success, and roll back on any exception. `init_db()` runs the `CREATE TABLE IF NOT
EXISTS` statements once at FastAPI startup (wired in via `app_factory.py`'s
`lifespan`).

#### `models/schemas.py` — the request/response contracts
Pydantic models that define exactly what shape of JSON the API accepts and returns —
`CustomerIn`, `TicketIn` (a claim/FNOL submission), `DraftOut`, `AcceptDraft`, etc.
FastAPI uses these to validate incoming requests automatically and to generate the
Swagger docs at `/docs`. `Literal[...]` fields (like `claim_type` and `priority`)
constrain inputs to a fixed set of allowed values.

#### `repositories/` — the data-access layer (SQL, and only SQL)
- **`customer_repo.py`** — `upsert_customer()` (insert-or-update by email, so
  re-registering the same person updates their record instead of duplicating it),
  `get_by_email()`, `get_by_id()`. Email is normalized to lowercase before every
  lookup so `Asha.Rao@Example.com` and `asha.rao@example.com` are the same customer.
- **`ticket_repo.py`** — `create_ticket()`, `get_ticket()`, `list_tickets()`,
  `count_open_for_customer()` (used by the "open claim load" tool), `set_status()`.
- **`draft_repo.py`** — `create_draft()`, `get_draft()`, `latest_for_ticket()`,
  `update_draft()`. Drafts store their `context_used` as a JSON blob in a TEXT column,
  which this file serializes/deserializes so callers always get a Python dict back.

None of these files know what an LLM, embedding, or memory is. They only know SQL.

#### `integrations/` — everything external

- **`llm/openai_client.py`** — creates and caches (`@lru_cache`, so it's a singleton) a
  `ChatOpenAI` instance for chat completions and an `OpenAIEmbeddings` instance for
  embeddings. Both use the same OpenAI API key. Raises immediately with a clear error
  if `OPENAI_API_KEY` is missing.

- **`rag/chroma_kb.py`** — the knowledge-base search engine. `ingest_knowledge_base()`
  reads every `.md` file in `knowledge_base/`, splits each into ~900-character chunks
  (using `RecursiveCharacterTextSplitter`, preferring to break at `##` headings so a
  rule is never cut in half), embeds them in batches, and `upsert`s them into a
  persistent ChromaDB collection on disk (`data/chroma/`). Using `upsert` with
  deterministic IDs (`filename::chunk-index`) makes re-running ingestion after editing
  a doc safe — it overwrites instead of duplicating. `search_knowledge()` embeds the
  query the same way and returns the top-k nearest chunks with their source filename
  and similarity score, wrapped in a try/except so a search failure never crashes draft
  generation — it just returns an empty list.

- **`memory/langmem_store.py`** — the long-term memory system, built on LangGraph's
  `InMemoryStore` (an in-RAM, optionally vector-indexed key-value store). This is where
  the "getting smarter over time" behavior lives. Two important ideas:
  - **Namespaces**: every memory is written to *two* separate scopes — one keyed by the
    individual customer's email, one keyed by their company. A brand-new claimant at a
    company with history still benefits from their colleagues' past resolutions.
  - **Layered fallbacks**: `search_memories()` tries a semantic vector search first; if
    that returns nothing (or embeddings are disabled/unavailable), it falls back to
    listing the most recent memories in that namespace instead. Every failure is caught
    and logged rather than raised, because memory is meant to be an *enhancement*, not
    a hard dependency — a memory outage should never stop a draft from being produced.

- **`tools/support_tools.py`** — two `@tool`-decorated functions the AI agent can
  choose to call: `lookup_customer_plan(email)` (returns the policyholder's plan tier
  and the SLA that applies) and `lookup_open_ticket_load(email)` (returns how many
  claims they currently have open, flagging possible duplicate filings above three).
  The model decides *whether* and *when* to call these purely by reading their
  docstrings, which LangChain sends to the API as the tool's description — so the
  docstrings are written like clear instructions to a colleague, not like internal
  comments.

#### `services/` — the orchestration / business-logic layer

- **`copilot_service.py`** — the heart of the system. `generate_recommendation(ticket,
  customer)` runs five steps:
  1. `_gather_context()` — queries memory and knowledge search **independently**, each
     wrapped in its own try/except, so one failing doesn't take down the other.
  2. `_format_context()` — turns the raw hits into labelled, readable text blocks.
  3. `_build_user_prompt()` — assembles the claim details + formatted context into one
     message.
  4. `_run_agent()` — hands the prompt, the two support tools, and a strict
     `SYSTEM_PROMPT` (which encodes every guardrail: ground claims in context, never
     state a settlement figure or liability decision, cite knowledge sources, output
     exactly four fixed sections, stay under 300 words) to a LangChain/LangGraph
     `create_agent`. This runs the model↔tool loop automatically — the agent decides
     for itself whether to call `lookup_customer_plan` or `lookup_open_ticket_load`
     before writing.
  5. If the agent produces no usable text (rare, but happens with some model/tool-loop
     combinations), `_fallback_generate()` makes one plain LLM call with no tools as a
     safety net. If even that fails, a static placeholder draft is returned so the API
     never 500s on the adjuster.

  Every memory hit, knowledge chunk, and tool call made along the way is captured into
  a `context_used` dictionary and returned alongside the draft — this is the exact data
  the Streamlit "Context used by the copilot" panel renders, giving the adjuster full
  transparency into *why* the AI wrote what it wrote.

- **`draft_service.py`** — manages the lifecycle around a draft:
  - `generate_draft_for_ticket()` — fetches the ticket + customer, calls
    `generate_recommendation()`, and persists the result. Designed to be safe to run as
    a FastAPI `BackgroundTask` (catches everything, just logs and returns `None` on
    failure rather than crashing the background thread).
  - `accept_draft()` — saves the (possibly human-edited) final text, marks the ticket
    `resolved`, and — **only if the adjuster left the "save to memory" checkbox
    ticked** — writes the approved resolution into memory via `mem.save_memory()`. This
    single conditional is what keeps low-quality or wrong drafts out of memory forever.
  - `discard_draft()` — marks the draft `discarded`. Nothing is written to memory.

#### `api/` — the HTTP layer

- **`app_factory.py`** — `create_app()` builds the `FastAPI` instance, registers a
  `lifespan` hook that calls `init_db()` once at startup, adds a permissive CORS
  middleware (fine for local development; the README explicitly flags this as
  something to lock down before any real deployment), and mounts every router.

- **`routers/health.py`** — `GET /health`: reports whether the app is up, which model
  is configured, and whether semantic (embedding-based) mode is active. This is what
  the Streamlit sidebar checks first, before showing anything else.

- **`routers/tickets.py`** — `POST /tickets` upserts the customer, creates the ticket,
  and immediately schedules `generate_draft_for_ticket` as a background task before
  returning `201` — the adjuster is never blocked waiting ~10-15 seconds for the LLM.
  Also `GET /tickets` (list) and `GET /tickets/{id}` (fetch one).

- **`routers/drafts.py`** — `POST /drafts/generate/{ticket_id}` (generate
  synchronously, used by the "Generate / regenerate" button), `GET
  /drafts/ticket/{ticket_id}` (latest draft for a claim), `PATCH /drafts/{id}` (save an
  edit without approving), `POST /drafts/{id}/accept`, `POST /drafts/{id}/discard`.

- **`routers/knowledge.py`** — `POST /knowledge/ingest` (re-chunk/re-embed everything
  in `knowledge_base/` — idempotent, safe to click repeatedly), `GET /knowledge/stats`
  (indexed chunk count), `GET /knowledge/search` (raw retrieval, useful for debugging
  what the RAG system actually returns for a given query).

- **`routers/memory.py`** — `GET /memory/search` — lets you probe both memory scopes
  directly for a given email/company/query, independent of drafting a claim. This
  backs the "Claim history" tab in the dashboard.

---

## 5. `knowledge_base/` — the policy rulebook

Plain Markdown files, each covering one policy topic — e.g.
`insurance-auto-coverage-and-deductible-guidelines.md`,
`insurance-claims-fraud-risk-indicators.md`,
`insurance-auto-claims-fnol-intake-checklist.md`. These are treated as the ground
truth the AI is allowed to cite. They are **not** automatically indexed — someone has
to click "Ingest knowledge base" (or call `POST /knowledge/ingest`) after adding or
editing a file, because re-embedding on every server restart would be slow and cost
money for documents that rarely change. Edit these files to change what "policy" the
copilot knows about; the (unrelated-looking) banking-topic files alongside them
(`banking-atm-cash-withdrawal-faq.md`, etc.) are extra sample knowledge content — every
`.md` file in this folder gets ingested the same way, regardless of topic.

---

## 6. `data/` — generated, local-only state

- `app.db` — the SQLite database file (customers, tickets, drafts).
- `chroma/` — ChromaDB's on-disk vector index for the knowledge base.

This whole folder is gitignored. Delete it (or just `data/chroma/`) and restart if you
ever change `EMBEDDING_MODEL` or `EMBEDDING_DIMS` — vectors of different widths can't
coexist in the same collection.

---

## 7. Key concepts glossary

| Term | Meaning in this project |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Searching a document store for relevant text and feeding it to the LLM as context, instead of relying on what the model "remembers." Here: `chroma_kb.py` searching `knowledge_base/`. |
| **Embedding** | Converting text into a list of numbers (a vector) that captures its meaning, so "similar meaning" texts end up as "nearby vectors." Used for both knowledge search and memory search. |
| **Vector database** | A database optimized for finding the nearest vectors to a query vector. ChromaDB, here. |
| **LangGraph** | The library providing `create_agent` (runs the model↔tool loop) and `InMemoryStore` (the memory backend). |
| **LangMem** | A library built on LangGraph's store that adds a managed, namespaced "memory" abstraction on top of raw key-value storage. |
| **Tool calling** | Giving an LLM a list of Python functions it can choose to invoke mid-conversation (here: the two functions in `support_tools.py`) to fetch live facts it couldn't otherwise know. |
| **Namespace** (in LangMem) | A tuple like `("claims", "customer", "asha-rao-example-com")` that scopes memories so a search only returns memories relevant to that customer or company. |
| **Draft lifecycle** | pending → (edited any number of times) → accepted **or** discarded. Only `accepted` + "save to memory" checked ever reaches long-term memory. |
| **`context_used`** | The JSON record, stored with every draft, of exactly which memories, knowledge chunks, and tool calls were used to write it — the transparency trail shown in the dashboard. |

---

## 8. How to actually run it

Two processes, two terminals:

```bash
# Terminal 1 — the API
uv run uvicorn customer_support_agent.main:app --reload --port 8000

# Terminal 2 — the dashboard
uv run streamlit run app.py
```

Then: open `http://localhost:8501`, click **Ingest knowledge base** in the sidebar
(only needed once, or after editing `knowledge_base/*.md`), register a claim, generate
its draft, read the **Context used by the copilot** panel, edit if you want, and
approve. Register a second, similar claim afterward and watch the new draft change —
that's the memory feedback loop working.

Full setup/config details (installing `uv`, `.env` variables, the complete API
reference) are in `README.md` — this guide is the map; `README.md` is the reference
manual for exact commands and parameters.

---

## 9. A note on `road_map.md` vs. the real code

`road_map.md` is a teaching document — a 12-phase, checkpoint-driven tutorial for
building this exact system from an empty folder. It's genuinely useful for
understanding *why* each file was introduced and in what order (settings → database →
FastAPI skeleton → RAG → memory → tools → orchestration → draft lifecycle →
dashboard...). One thing to know: it was drafted using Groq for chat and Google Gemini
for embeddings, but the project as actually built (see `pyproject.toml`,
`settings.py`, and every `integrations/` file) uses **OpenAI for both**. If the two
ever disagree on a detail, trust the real source files and `README.md` — `road_map.md`
is history/teaching material, not the current spec.
