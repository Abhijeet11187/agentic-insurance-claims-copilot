# Insurance Claims Support AI Copilot — Implementation Roadmap

A complete, beginner-friendly build guide for the system in the architecture diagram:
**FastAPI + Streamlit + LangGraph/LangChain + LangMem + ChromaDB + OpenAI + SQLite**.

Follow the phases in order. Each phase has: **Goal → Files you create → Code → Checkpoint**.
Do not move to the next phase until the checkpoint passes.

---

## Table of Contents

| Phase | What you build | Time |
|---|---|---|
| [0](#phase-0--understand-the-system-before-you-code) | Mental model of the system | 20 min |
| [1](#phase-1--project-bootstrap) | Folder structure, `uv`, `.env`, settings | 45 min |
| [2](#phase-2--data-layer-sqlite--repositories) | SQLite tables + repositories | 45 min |
| [3](#phase-3--fastapi-skeleton) | App factory, routers, health check | 40 min |
| [4](#phase-4--knowledge-base--rag-with-chromadb) | Knowledge base + ChromaDB RAG | 60 min |
| [5](#phase-5--langmem-memory-layer) | LangMem long-term memory | 75 min |
| [6](#phase-6--structured-tool-calling-layer) | `lookup_customer_plan`, `lookup_open_ticket_load` | 30 min |
| [7](#phase-7--copilot-orchestration-the-brain) | Memory + RAG + tools + OpenAI → draft | 90 min |
| [8](#phase-8--draft-lifecycle--human-in-the-loop) | Generate / edit / approve / discard | 60 min |
| [9](#phase-9--streamlit-dashboard) | Operator UI | 75 min |
| [10](#phase-10--testing-with-pytest) | Pytest suite | 45 min |
| [11](#phase-11--docker--docker-compose) | Containerization | 45 min |
| [12](#phase-12--cicd--ec2-deployment) | GitHub Actions + EC2 | 60 min |

---

## Phase 0 — Understand the system before you code

### The one-sentence version

> A human claims adjuster registers a claim → the AI reads *past similar claims* (memory), *company policy docs* (RAG), and *live operational facts* (tools) → it writes a **draft** recommendation → the human edits and approves it → the approved text is **written back into memory** so the next claim is handled better.

### The request flow, in plain text

```
Adjuster types a claim in Streamlit
        │
        ▼
POST /tickets  ──────────────►  FastAPI
                                   │
                                   ├─► SQLite: save customer + ticket
                                   │
                                   └─► BackgroundTask: generate_draft()
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
      1. MEMORY RETRIEVAL           2. KNOWLEDGE RETRIEVAL          3. TOOL CALLING
      LangMem + InMemoryStore       ChromaDB (top-k chunks)         plan lookup,
      "how did we resolve                                          open-ticket load
       claims like this before?"
              │                              │                              │
              └──────────────────────────────┼──────────────────────────────┘
                                             ▼
                              4. PROMPT COMPOSITION (system + user)
                                             ▼
                              5. OPENAI LLM  →  draft text
                                             ▼
                              6. Save draft + context_used to SQLite
                                             │
        ┌────────────────────────────────────┘
        ▼
Adjuster reviews in Streamlit → edits → APPROVE
        ▼
POST /drafts/{id}/accept  →  writes approved resolution into LangMem
```

### The four "context sources" — know the difference

| Source | Answers the question | Storage | Fresh or stale? |
|---|---|---|---|
| **Memory (LangMem)** | "What did *we* decide on similar past claims?" | LangGraph `InMemoryStore` | Grows over time |
| **RAG (ChromaDB)** | "What does the *policy document* say?" | Chroma vectors on disk | Static until re-ingested |
| **Tools** | "What is *true right now*?" (plan tier, open tickets) | SQLite, live query | Always live |
| **SQLite records** | "What are the raw claim fields?" | SQLite | Always live |

A beginner mistake is to shove everything into RAG. Don't. **RAG = documents. Memory = experience. Tools = live facts.**

### Prerequisites checklist

- [ ] Python 3.11 installed (`python --version`)
- [ ] `uv` installed → `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] An **OpenAI API key** with billing enabled → platform.openai.com/api-keys
- [ ] Docker Desktop (needed only from Phase 11)
- [ ] A GitHub account (needed only from Phase 12)

**One key powers three things:** chat completion (`gpt-4.1-mini`), knowledge-base embeddings, and memory embeddings (`text-embedding-3-small`). That is the main advantage of going all-OpenAI — one provider, one bill, one failure domain.

> **Keep the degraded path anyway.** The config below has an `ENABLE_SEMANTIC` switch. Turn it off and the system still runs: RAG falls back to Chroma's built-in embedding function and memory falls back to "list recent memories" instead of semantic search. Build and test that path deliberately — it is one of the strongest things you can talk about in an interview, and it is what saves you when the embeddings endpoint rate-limits you mid-demo.

### A note on model choice

| Setting | Default used here | Why |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4.1-mini` | Cheap, fast, reliable tool calling. Swap to `gpt-4.1` or a `gpt-5`-class model for higher quality; check platform.openai.com/docs/models for what is current |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dims, very cheap, more than enough for a 5-file knowledge base. `text-embedding-3-large` (3072 dims) if you need better recall |

> **If you switch to a reasoning model** (the `o`-series or `gpt-5` thinking variants), remove `temperature` from the client — those endpoints reject it. Everything else in this guide is unchanged.

> **Changing the embedding model later is a breaking change.** Vectors of different dimensions cannot live in the same Chroma collection or the same LangGraph store index. If you change it, delete `data/chroma/`, re-ingest, and restart the API so the memory store is rebuilt.

### What this will cost you to build

Rough orders of magnitude, not a quote — check current pricing before you start.

| Operation | Roughly | Notes |
|---|---|---|
| Ingest 5 knowledge files once | a fraction of a cent | ~60–100 chunks, embeddings only |
| One draft generation | well under a cent with `gpt-4.1-mini` | Retrieval + tool loop + ~300-word draft |
| Full 5-day build with heavy testing | a couple of dollars | The agent tool loop makes 2–3 model calls per draft, so watch that |

Two habits that keep the bill near zero: set a **hard usage limit** in the OpenAI dashboard on day one, and keep `ENABLE_SEMANTIC=false` while you are debugging non-AI code so no embedding calls fire.

---

## Phase 1 — Project bootstrap

**Goal:** a clean, importable package with centralized configuration.

### 1.1 Create the project

```bash
mkdir insurance-claims-copilot && cd insurance-claims-copilot
uv init --python 3.11
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 1.2 Install dependencies

```bash
uv add fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv
uv add langchain langchain-core langgraph langchain-openai
uv add langmem
uv add chromadb langchain-text-splitters
uv add streamlit requests
uv add --dev pytest pytest-asyncio httpx ruff
```

### 1.3 Create the folder structure

```bash
mkdir -p customer_support_agent/{core,db,models,repositories,services,api/routers}
mkdir -p customer_support_agent/integrations/{rag,memory,llm,tools}
mkdir -p knowledge_base data tests
find customer_support_agent -type d -exec touch {}/__init__.py \;
touch tests/__init__.py
```

Final layout you are working toward:

```
insurance-claims-copilot/
├── customer_support_agent/
│   ├── core/settings.py                     # Phase 1
│   ├── db/database.py                       # Phase 2
│   ├── models/schemas.py                    # Phase 2
│   ├── repositories/
│   │   ├── customer_repo.py                 # Phase 2
│   │   ├── ticket_repo.py                   # Phase 2
│   │   └── draft_repo.py                    # Phase 2
│   ├── integrations/
│   │   ├── rag/chroma_kb.py                 # Phase 4
│   │   ├── memory/langmem_store.py          # Phase 5
│   │   ├── tools/support_tools.py           # Phase 6
│   │   └── llm/openai_client.py             # Phase 7
│   ├── services/
│   │   ├── copilot_service.py               # Phase 7
│   │   └── draft_service.py                 # Phase 8
│   ├── api/
│   │   ├── app_factory.py                   # Phase 3
│   │   └── routers/{health,customers,tickets,drafts,knowledge}.py
│   └── main.py                              # Phase 3
├── knowledge_base/*.md                      # Phase 4
├── app.py                                   # Phase 9  (Streamlit)
├── tests/                                   # Phase 10
├── Dockerfile / docker-compose.yml          # Phase 11
└── .github/workflows/ci.yml                 # Phase 12
```

### 1.4 Environment file

Create `.env`:

```bash
OPENAI_API_KEY=sk-proj-your_key_here
OPENAI_MODEL=gpt-4.1-mini

# Embeddings — used by BOTH the RAG layer and the memory layer
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536

# Set to false to run fully offline-degraded (no embedding calls at all)
ENABLE_SEMANTIC=true

APP_ENV=local
API_BASE_URL=http://localhost:8000
```

And `.env.example` (same file with values blanked — commit this one, never `.env`).

Create `.gitignore`:

```
.venv/
__pycache__/
*.pyc
.env
data/
.pytest_cache/
```

### 1.5 Centralized settings

**`customer_support_agent/core/settings.py`**

```python
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# settings.py -> core -> customer_support_agent -> project root
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Insurance Claims Support Copilot"
    app_env: str = "local"
    api_base_url: str = "http://localhost:8000"

    # --- LLM (OpenAI) ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.2
    llm_timeout: int = 60
    llm_max_retries: int = 3

    # --- Embeddings (OpenAI, same key) ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    embedding_batch_size: int = 64
    enable_semantic: bool = True      # flip to False to exercise the fallback path

    # --- Paths ---
    data_dir: Path = BASE_DIR / "data"
    db_path: Path = BASE_DIR / "data" / "app.db"
    chroma_path: Path = BASE_DIR / "data" / "chroma"
    knowledge_base_dir: Path = BASE_DIR / "knowledge_base"

    # --- RAG ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    rag_top_k: int = 4

    # --- Memory ---
    memory_top_k: int = 4

    @property
    def semantic_enabled(self) -> bool:
        """Embeddings use the same OpenAI key as the chat model."""
        return self.enable_semantic and bool(self.openai_api_key.strip())

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
```

> **Why `@lru_cache`?** Settings are read from disk once and reused everywhere. Without it, every import re-parses `.env`.

### ✅ Checkpoint 1

```bash
python -c "from customer_support_agent.core.settings import get_settings; s=get_settings(); print(s.app_name, s.openai_model, s.db_path, s.semantic_enabled)"
```

Expected: the app name, `gpt-4.1-mini`, an absolute `data/app.db` path, and `True`. The `data/` folder now exists.

---

## Phase 2 — Data layer (SQLite + repositories)

**Goal:** three tables and a thin data-access layer. No business logic here.

### 2.1 Database connection + schema

**`customer_support_agent/db/database.py`**

```python
import sqlite3
from contextlib import contextmanager
from typing import Iterator

from customer_support_agent.core.settings import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    company     TEXT,
    plan_tier   TEXT DEFAULT 'standard',
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tickets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL,
    subject      TEXT NOT NULL,
    body         TEXT NOT NULL,
    claim_type   TEXT DEFAULT 'auto',
    priority     TEXT DEFAULT 'medium',
    status       TEXT DEFAULT 'open',
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE IF NOT EXISTS drafts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id     INTEGER NOT NULL,
    content       TEXT NOT NULL,
    status        TEXT DEFAULT 'pending',   -- pending | accepted | discarded
    context_used  TEXT DEFAULT '{}',        -- JSON blob for transparency
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticket_id) REFERENCES tickets (id)
);
"""


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
```

> **`check_same_thread=False`** is required because FastAPI runs handlers on a thread pool. Opening a fresh connection per request (as above) keeps this safe.

### 2.2 Pydantic schemas

**`customer_support_agent/models/schemas.py`**

```python
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class CustomerIn(BaseModel):
    name: str
    email: EmailStr
    company: Optional[str] = None
    plan_tier: Literal["basic", "standard", "premium"] = "standard"


class CustomerOut(CustomerIn):
    id: int


class TicketIn(BaseModel):
    """A First Notice of Loss (FNOL) / claim intake."""
    customer: CustomerIn
    subject: str = Field(..., min_length=3)
    body: str = Field(..., min_length=10)
    claim_type: Literal["auto", "property", "health", "other"] = "auto"
    priority: Literal["low", "medium", "high"] = "medium"


class TicketOut(BaseModel):
    id: int
    customer_id: int
    subject: str
    body: str
    claim_type: str
    priority: str
    status: str
    created_at: str


class DraftOut(BaseModel):
    id: int
    ticket_id: int
    content: str
    status: str
    context_used: dict[str, Any] = {}
    created_at: str
    updated_at: str


class DraftUpdate(BaseModel):
    content: str


class AcceptDraft(BaseModel):
    content: Optional[str] = None       # final edited text
    save_to_memory: bool = True         # human controls memory writes
```

### 2.3 Repositories

**`customer_support_agent/repositories/customer_repo.py`**

```python
from typing import Optional

from customer_support_agent.db.database import get_conn


def normalize_email(email: str) -> str:
    return email.strip().lower()


def upsert_customer(
    name: str, email: str, company: Optional[str], plan_tier: str
) -> dict:
    email = normalize_email(email)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO customers (name, email, company, plan_tier)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                company = excluded.company,
                plan_tier = excluded.plan_tier
            """,
            (name, email, company, plan_tier),
        )
        row = conn.execute(
            "SELECT * FROM customers WHERE email = ?", (email,)
        ).fetchone()
    return dict(row)


def get_by_email(email: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE email = ?", (normalize_email(email),)
        ).fetchone()
    return dict(row) if row else None


def get_by_id(customer_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
    return dict(row) if row else None
```

**`customer_support_agent/repositories/ticket_repo.py`**

```python
from typing import Optional

from customer_support_agent.db.database import get_conn


def create_ticket(
    customer_id: int, subject: str, body: str, claim_type: str, priority: str
) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO tickets (customer_id, subject, body, claim_type, priority)
               VALUES (?, ?, ?, ?, ?)""",
            (customer_id, subject, body, claim_type, priority),
        )
        row = conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def get_ticket(ticket_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
    return dict(row) if row else None


def list_tickets(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_open_for_customer(customer_id: int) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM tickets WHERE customer_id = ? AND status = 'open'",
            (customer_id,),
        ).fetchone()
    return int(row["c"])


def set_status(ticket_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id)
        )
```

**`customer_support_agent/repositories/draft_repo.py`**

```python
import json
from typing import Any, Optional

from customer_support_agent.db.database import get_conn


def _row_to_draft(row) -> dict:
    d = dict(row)
    try:
        d["context_used"] = json.loads(d.get("context_used") or "{}")
    except json.JSONDecodeError:
        d["context_used"] = {}
    return d


def create_draft(ticket_id: int, content: str, context_used: dict[str, Any]) -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO drafts (ticket_id, content, context_used) VALUES (?, ?, ?)",
            (ticket_id, content, json.dumps(context_used, default=str)),
        )
        row = conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return _row_to_draft(row)


def get_draft(draft_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    return _row_to_draft(row) if row else None


def latest_for_ticket(ticket_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM drafts WHERE ticket_id = ? ORDER BY id DESC LIMIT 1",
            (ticket_id,),
        ).fetchone()
    return _row_to_draft(row) if row else None


def update_draft(draft_id: int, content: Optional[str], status: Optional[str]) -> dict:
    with get_conn() as conn:
        if content is not None:
            conn.execute(
                "UPDATE drafts SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (content, draft_id),
            )
        if status is not None:
            conn.execute(
                "UPDATE drafts SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, draft_id),
            )
        row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    return _row_to_draft(row)
```

### ✅ Checkpoint 2

```bash
python - <<'PY'
from customer_support_agent.db.database import init_db
from customer_support_agent.repositories import customer_repo, ticket_repo

init_db()
c = customer_repo.upsert_customer("Asha Rao", "Asha.Rao@Example.com", "Acme Ltd", "premium")
t = ticket_repo.create_ticket(c["id"], "Rear-end collision", "Hit from behind at a signal.", "auto", "high")
print(c)
print(t)
print("open tickets:", ticket_repo.count_open_for_customer(c["id"]))
PY
```

Expected: customer with lowercased email, a ticket row, `open tickets: 1`.

---

## Phase 3 — FastAPI skeleton

**Goal:** a running API with health check and customer/ticket endpoints. No AI yet.

### 3.1 App factory

**`customer_support_agent/api/app_factory.py`**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from customer_support_agent.core.settings import get_settings
from customer_support_agent.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                    # runs once at startup
    yield
    # add cleanup here if needed


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],     # tighten for production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from customer_support_agent.api.routers import (
        drafts, health, knowledge, tickets,
    )

    app.include_router(health.router)
    app.include_router(tickets.router)
    app.include_router(drafts.router)
    app.include_router(knowledge.router)
    return app
```

**`customer_support_agent/main.py`**

```python
from customer_support_agent.api.app_factory import create_app

app = create_app()
```

### 3.2 Routers

**`customer_support_agent/api/routers/health.py`**

```python
from fastapi import APIRouter

from customer_support_agent.core.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "app": s.app_name,
        "llm_configured": bool(s.openai_api_key),
        "model": s.openai_model,
        "semantic_enabled": s.semantic_enabled,
    }
```

**`customer_support_agent/api/routers/tickets.py`**

```python
from fastapi import APIRouter, BackgroundTasks, HTTPException

from customer_support_agent.models.schemas import TicketIn, TicketOut
from customer_support_agent.repositories import customer_repo, ticket_repo

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=TicketOut, status_code=201)
def create_ticket(payload: TicketIn, background: BackgroundTasks):
    customer = customer_repo.upsert_customer(
        payload.customer.name,
        payload.customer.email,
        payload.customer.company,
        payload.customer.plan_tier,
    )
    ticket = ticket_repo.create_ticket(
        customer["id"], payload.subject, payload.body,
        payload.claim_type, payload.priority,
    )

    # Phase 8 wires the copilot in here:
    # from customer_support_agent.services.draft_service import generate_draft_for_ticket
    # background.add_task(generate_draft_for_ticket, ticket["id"])

    return ticket


@router.get("", response_model=list[TicketOut])
def list_tickets(limit: int = 50):
    return ticket_repo.list_tickets(limit)


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: int):
    ticket = ticket_repo.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket
```

Create **placeholder** files so imports don't break — you fill them in Phases 4 and 8:

**`customer_support_agent/api/routers/knowledge.py`**

```python
from fastapi import APIRouter

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
```

**`customer_support_agent/api/routers/drafts.py`**

```python
from fastapi import APIRouter

router = APIRouter(prefix="/drafts", tags=["drafts"])
```

### ✅ Checkpoint 3

```bash
uvicorn customer_support_agent.main:app --reload --port 8000
```

Open `http://localhost:8000/docs`. Create a ticket from the Swagger UI, then `GET /tickets`. You should see it.

---

## Phase 4 — Knowledge base + RAG with ChromaDB

**Goal:** turn Markdown policy docs into a searchable vector index.

### 4.1 Write the knowledge files

Create these five files in `knowledge_base/`. Real content matters — the LLM can only be as grounded as your documents. Keep each 300–800 words.

```
knowledge_base/
├── insurance-auto-claims-fnol-intake-checklist.md
├── insurance-auto-coverage-and-deductible-guidelines.md
├── insurance-auto-required-documents-by-claim-type.md
├── insurance-claims-settlement-sla-and-communication.md
└── insurance-claims-fraud-risk-indicators.md
```

Starter for the first one (write the other four in the same style):

```markdown
# FNOL Intake Checklist — Auto Claims

## Mandatory fields at first notice
- Policy number and policyholder name
- Date, time, and location of loss
- Description of the incident in the claimant's own words
- Whether the vehicle is drivable
- Injuries reported (yes / no / unknown)
- Police report number, if one was filed
- Third-party details: name, contact, insurer, vehicle registration

## Intake rules
- If injuries are reported, escalate to the bodily-injury queue within 2 hours.
- If the vehicle is not drivable, arrange towing and a rental under the policy's
  transportation benefit before closing the intake call.
- Never quote a settlement figure at FNOL. Only confirm that the claim is registered.

## Common intake gaps
- Missing third-party insurer details is the single most common cause of delay.
- Photographs taken at the scene should be requested during the first contact.
```

### 4.2 The RAG module

**`customer_support_agent/integrations/rag/chroma_kb.py`**

```python
from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_support_agent.core.settings import get_settings

logger = logging.getLogger(__name__)
COLLECTION_NAME = "insurance_knowledge"

_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        s = get_settings()
        _client = chromadb.PersistentClient(path=str(s.chroma_path))
    return _client


def _get_embedder():
    """Return the shared OpenAI embeddings client, or None for Chroma's default."""
    if not get_settings().semantic_enabled:
        return None
    from customer_support_agent.integrations.llm.openai_client import get_embeddings

    return get_embeddings()


def _get_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def ingest_knowledge_base() -> dict[str, Any]:
    """Read every .md file, chunk it, embed it, and upsert into Chroma."""
    s = get_settings()
    files = sorted(s.knowledge_base_dir.glob("*.md"))
    if not files:
        return {"files": 0, "chunks": 0, "message": "knowledge_base/ is empty"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )

    ids, docs, metas = [], [], []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{path.stem}::{i}")
            docs.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    collection = _get_collection()
    embedder = _get_embedder()

    # Batch to stay under OpenAI's per-request token limit for embeddings.
    BATCH = s.embedding_batch_size
    for start in range(0, len(docs), BATCH):
        b_ids = ids[start : start + BATCH]
        b_docs = docs[start : start + BATCH]
        b_metas = metas[start : start + BATCH]
        kwargs: dict[str, Any] = {
            "ids": b_ids, "documents": b_docs, "metadatas": b_metas
        }
        if embedder is not None:
            kwargs["embeddings"] = embedder.embed_documents(b_docs)
        collection.upsert(**kwargs)

    return {
        "files": len(files),
        "chunks": len(docs),
        "semantic": embedder is not None,
    }


def search_knowledge(query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
    """Return the top-k most relevant knowledge chunks with source metadata."""
    s = get_settings()
    top_k = top_k or s.rag_top_k
    collection = _get_collection()

    if collection.count() == 0:
        return []

    try:
        embedder = _get_embedder()
        if embedder is not None:
            res = collection.query(
                query_embeddings=[embedder.embed_query(query)], n_results=top_k
            )
        else:
            res = collection.query(query_texts=[query], n_results=top_k)
    except Exception as exc:                       # never break draft generation
        logger.warning("Knowledge search failed: %s", exc)
        return []

    hits = []
    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "score": round(1 - float(dist), 4),   # cosine distance -> similarity
            }
        )
    return hits


def knowledge_stats() -> dict[str, Any]:
    return {
        "collection": COLLECTION_NAME,
        "chunks": _get_collection().count(),
        "semantic": _get_embedder() is not None,
    }
```

> **Why `upsert` and not `add`?** Re-running ingestion with `add` throws a duplicate-ID error. `upsert` makes ingestion **idempotent** — you can press the button in the dashboard as many times as you like.

### 4.3 Knowledge router

Replace **`customer_support_agent/api/routers/knowledge.py`**:

```python
from fastapi import APIRouter, Query

from customer_support_agent.integrations.rag import chroma_kb

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/ingest")
def ingest():
    return chroma_kb.ingest_knowledge_base()


@router.get("/stats")
def stats():
    return chroma_kb.knowledge_stats()


@router.get("/search")
def search(q: str = Query(..., min_length=3), top_k: int = 4):
    return {"query": q, "hits": chroma_kb.search_knowledge(q, top_k)}
```

### ✅ Checkpoint 4

```bash
curl -X POST http://localhost:8000/knowledge/ingest
curl "http://localhost:8000/knowledge/search?q=what%20documents%20are%20needed%20for%20a%20collision%20claim"
```

Expected: ingest reports a chunk count > 0; search returns hits with `source` filenames and scores. **If the top hit is obviously unrelated, your knowledge files are too thin — go back and write more content.**

---

## Phase 5 — LangMem memory layer

**Goal:** persistent, scoped, semantic memory of past claim resolutions.

### 5.1 The concepts (read this before coding)

- **`InMemoryStore`** — LangGraph's key-value store with optional vector indexing. Data lives in RAM; it disappears on restart. That is acceptable for this project, and swapping it for a Postgres store later is a one-line change.
- **Namespace** — a tuple that scopes memories, e.g. `("claims", "customer", "asha.rao@example.com")`. Two scopes are used:
  - **customer scope** — normalized email
  - **company scope** — `company::<slug>` so teammates' resolutions are shared
- **`create_manage_memory_tool`** — LangMem's tool for writing memories. Using the tool (instead of raw `store.put`) means the *agent itself* can be given memory-writing ability later.

### 5.2 The memory module

**`customer_support_agent/integrations/memory/langmem_store.py`**

```python
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional

from langgraph.store.memory import InMemoryStore

from customer_support_agent.core.settings import get_settings

logger = logging.getLogger(__name__)

_store: Optional[InMemoryStore] = None
ROOT = "claims"


# ---------------------------------------------------------------- namespaces
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def company_slug(company: Optional[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (company or "general").strip().lower()).strip("-")
    return f"company::{slug or 'general'}"


def customer_namespace(email: str) -> tuple[str, ...]:
    return (ROOT, "customer", normalize_email(email))


def company_namespace(company: Optional[str]) -> tuple[str, ...]:
    return (ROOT, "company", company_slug(company))


# -------------------------------------------------------------------- store
def get_store() -> InMemoryStore:
    """Singleton store. Vector-indexed when semantic mode is enabled."""
    global _store
    if _store is not None:
        return _store

    s = get_settings()
    if s.semantic_enabled:
        try:
            from customer_support_agent.integrations.llm.openai_client import (
                get_embeddings,
            )

            embedder = get_embeddings()
            _store = InMemoryStore(
                index={
                    "dims": s.embedding_dims,
                    "embed": lambda texts: embedder.embed_documents(list(texts)),
                }
            )
            logger.info("Memory store initialised with semantic index")
            return _store
        except Exception as exc:
            logger.warning("Semantic memory unavailable (%s); using plain store", exc)

    _store = InMemoryStore()          # non-semantic fallback
    return _store


def semantic_available() -> bool:
    return getattr(get_store(), "index_config", None) is not None


# -------------------------------------------------------------------- write
def save_memory(
    content: str,
    email: str,
    company: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write one resolution memory into both customer and company scopes."""
    store = get_store()
    payload = {"content": content.strip(), "metadata": metadata or {}}
    written = []

    for namespace in (customer_namespace(email), company_namespace(company)):
        key = str(uuid.uuid4())
        try:
            _write_via_langmem(store, namespace, payload)
        except Exception as exc:
            logger.warning("LangMem tool write failed (%s); using store.put", exc)
            store.put(namespace, key, payload)
        written.append("/".join(namespace))

    return {"saved": True, "namespaces": written}


def _write_via_langmem(store: InMemoryStore, namespace, payload: dict) -> None:
    """Preferred path: LangMem's managed memory tool."""
    from langmem import create_manage_memory_tool

    tool = create_manage_memory_tool(namespace=namespace, store=store)
    tool.invoke({"action": "create", "content": payload["content"]})


# --------------------------------------------------------------------- read
def search_memories(
    query: str,
    email: str,
    company: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Search customer scope then company scope; dedupe; fall back to recent."""
    store = get_store()
    limit = limit or get_settings().memory_top_k
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for scope, namespace in (
        ("customer", customer_namespace(email)),
        ("company", company_namespace(company)),
    ):
        for item in _search_one(store, namespace, query, limit):
            text = _extract_text(item.value)
            fingerprint = text[:160].strip().lower()
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            results.append(
                {
                    "memory": text,
                    "score": round(float(getattr(item, "score", 0.0) or 0.0), 4),
                    "metadata": {
                        "scope": scope,
                        "namespace": "/".join(namespace),
                        "key": item.key,
                    },
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def _search_one(store, namespace, query: str, limit: int) -> list:
    """Semantic search, then recent-listing fallback."""
    try:
        hits = store.search(namespace, query=query, limit=limit)
        if hits:
            return hits
    except Exception as exc:
        logger.warning("Semantic memory search failed (%s); listing recent", exc)

    try:
        return store.search(namespace, limit=limit)     # no query = recent items
    except Exception:
        return []


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "memory", "value"):
            if isinstance(value.get(key), str):
                return value[key]
        return str(value)
    return str(value)
```

### 5.3 Why every read path has a fallback

This is the single most important design idea in the memory layer, and the deck calls it out explicitly:

| Condition | Behaviour |
|---|---|
| Semantic on, memories exist, semantic hit | Ranked semantic results |
| Semantic on, no semantic match | Falls back to recent-memory listing |
| `ENABLE_SEMANTIC=false`, or no OpenAI key | Plain store, recent-memory listing only |
| OpenAI embeddings endpoint rate-limits or errors | Logged, plain store, recent-memory listing |
| Store or embedding call throws | Logged, returns `[]`, **draft generation continues** |

The copilot must never crash because memory was unavailable. Memory is an *enhancement*, not a dependency.

### ✅ Checkpoint 5

```bash
python - <<'PY'
from customer_support_agent.integrations.memory import langmem_store as mem

mem.save_memory(
    "Rear-end collision, third party at fault. Approved repair at network garage; "
    "deductible waived under no-fault clause. Settled in 6 days.",
    email="Asha.Rao@example.com", company="Acme Ltd",
    metadata={"claim_type": "auto", "ticket_id": 1},
)
for hit in mem.search_memories("rear ended, who pays the deductible?",
                               email="asha.rao@example.com", company="Acme Ltd"):
    print(hit["metadata"]["scope"], "|", round(hit["score"], 3), "|", hit["memory"][:70])
PY
```

Expected: at least one result with a real similarity score. Then set `ENABLE_SEMANTIC=false` in `.env` and run it again — you should still get the memory back, now with a `0.0` score from the recent-listing fallback. **Both runs must pass.** Set it back to `true` afterwards.

---

## Phase 6 — Structured tool-calling layer

**Goal:** give the agent live operational facts it cannot get from documents.

**`customer_support_agent/integrations/tools/support_tools.py`**

```python
from langchain_core.tools import tool

from customer_support_agent.repositories import customer_repo, ticket_repo

SLA_BY_TIER = {
    "premium": "4 business hours first response, 5 business days to settlement decision",
    "standard": "1 business day first response, 10 business days to settlement decision",
    "basic": "2 business days first response, 15 business days to settlement decision",
}


@tool
def lookup_customer_plan(email: str) -> str:
    """Look up a policyholder's plan tier and the SLA that applies to their claims.

    Use this whenever the response depends on entitlement, priority, or turnaround
    time. Pass the policyholder's email address.
    """
    customer = customer_repo.get_by_email(email)
    if not customer:
        return f"No policyholder found for {email}."
    tier = (customer.get("plan_tier") or "standard").lower()
    return (
        f"Policyholder: {customer['name']} ({customer['email']}). "
        f"Company: {customer.get('company') or 'N/A'}. "
        f"Plan tier: {tier}. SLA: {SLA_BY_TIER.get(tier, SLA_BY_TIER['standard'])}."
    )


@tool
def lookup_open_ticket_load(email: str) -> str:
    """Return how many claims this policyholder currently has open.

    Use this to judge whether to acknowledge existing open claims or to flag a
    possible duplicate filing.
    """
    customer = customer_repo.get_by_email(email)
    if not customer:
        return f"No policyholder found for {email}."
    count = ticket_repo.count_open_for_customer(customer["id"])
    if count == 0:
        return f"{customer['name']} has no other open claims."
    if count >= 3:
        return (
            f"{customer['name']} has {count} open claims — high load. "
            "Check for duplicate filings and consider consolidating updates."
        )
    return f"{customer['name']} has {count} open claim(s)."


SUPPORT_TOOLS = [lookup_customer_plan, lookup_open_ticket_load]
```

> **The docstring is the prompt.** The LLM decides whether to call a tool by reading its docstring. Vague docstring → the tool never gets called, or gets called at the wrong time. Write them as instructions to a new colleague.

### ✅ Checkpoint 6

```bash
python - <<'PY'
from customer_support_agent.integrations.tools.support_tools import (
    lookup_customer_plan, lookup_open_ticket_load,
)
print(lookup_customer_plan.invoke({"email": "asha.rao@example.com"}))
print(lookup_open_ticket_load.invoke({"email": "asha.rao@example.com"}))
print(lookup_customer_plan.name, "|", lookup_customer_plan.description[:60])
PY
```

---

## Phase 7 — Copilot orchestration (the brain)

**Goal:** one function that takes a ticket and returns `(draft_text, context_used)`.

### 7.1 OpenAI client (chat + embeddings in one place)

Both the RAG layer and the memory layer import `get_embeddings()` from here, so there is
exactly one embeddings client in the process and the model name is configured once.

**`customer_support_agent/integrations/llm/openai_client.py`**

```python
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from customer_support_agent.core.settings import get_settings


def _require_key() -> str:
    key = get_settings().openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    return key


@lru_cache
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_model,
        api_key=_require_key(),
        temperature=s.llm_temperature,
        timeout=s.llm_timeout,
        max_retries=s.llm_max_retries,
    )
    # Reasoning models (o-series, gpt-5 thinking) reject `temperature`.
    # If you switch to one, drop that line.


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=_require_key(),
        dimensions=s.embedding_dims,
        chunk_size=s.embedding_batch_size,
        max_retries=s.llm_max_retries,
    )
```

> **Why `dimensions` is passed explicitly:** `text-embedding-3-*` supports shortened
> output vectors. Pinning it means your Chroma collection and your LangGraph store
> index always agree on vector width — mismatched dims is the single most common
> runtime error when people swap embedding models mid-project.

### 7.2 The orchestration service

**`customer_support_agent/services/copilot_service.py`**

```python
from __future__ import annotations

import logging
from typing import Any

from customer_support_agent.core.settings import get_settings
from customer_support_agent.integrations.llm.openai_client import get_llm
from customer_support_agent.integrations.memory import langmem_store as mem
from customer_support_agent.integrations.rag import chroma_kb
from customer_support_agent.integrations.tools.support_tools import SUPPORT_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an internal AI copilot for insurance claims support agents.
You do NOT talk to the customer and you do NOT make final claim decisions.
You write a draft recommendation that a licensed human adjuster will review and edit.

Rules you must follow:
1. Ground every factual statement in the CONTEXT provided. If the context does not
   cover something, write "Needs verification: <what is missing>" instead of guessing.
2. Never promise, estimate, or imply a settlement amount.
3. Never state a final liability or coverage decision. Recommend, do not decide.
4. Use the operational tools available to you when plan tier, SLA, or the
   policyholder's open-claim load is relevant.
5. Cite the knowledge source filename in brackets when you rely on a policy document.

Output exactly these four sections in markdown:

**Summary** - two sentences on what happened.
**Recommended next steps** - 3 to 5 numbered actions for the adjuster.
**Information to request** - bullets, or "None" if nothing is missing.
**Risk / compliance notes** - bullets covering fraud indicators, SLA risk, or
escalation triggers. Write "None identified" if there are none.

Keep the whole draft under 300 words."""


# ------------------------------------------------------------------ context
def _gather_context(ticket: dict, customer: dict) -> dict[str, Any]:
    query = f"{ticket['subject']} {ticket['body']}"
    context: dict[str, Any] = {
        "memory_hits": [], "knowledge_hits": [], "tool_calls": [], "errors": [],
    }

    try:
        context["memory_hits"] = mem.search_memories(
            query, email=customer["email"], company=customer.get("company")
        )
    except Exception as exc:
        logger.exception("memory retrieval failed")
        context["errors"].append(f"memory: {exc}")

    try:
        context["knowledge_hits"] = chroma_kb.search_knowledge(query)
    except Exception as exc:
        logger.exception("knowledge retrieval failed")
        context["errors"].append(f"knowledge: {exc}")

    return context


def _format_context(context: dict[str, Any]) -> str:
    parts: list[str] = []

    if context["memory_hits"]:
        lines = [
            f"- [{h['metadata']['scope']} scope] {h['memory']}"
            for h in context["memory_hits"]
        ]
        parts.append("PAST RESOLUTIONS (memory):\n" + "\n".join(lines))
    else:
        parts.append("PAST RESOLUTIONS (memory): none found.")

    if context["knowledge_hits"]:
        lines = [f"- [{h['source']}] {h['text']}" for h in context["knowledge_hits"]]
        parts.append("POLICY KNOWLEDGE (retrieved):\n" + "\n".join(lines))
    else:
        parts.append("POLICY KNOWLEDGE (retrieved): none found.")

    return "\n\n".join(parts)


def _build_user_prompt(ticket: dict, customer: dict, context: dict) -> str:
    return f"""CLAIM DETAILS
Ticket ID: {ticket['id']}
Claim type: {ticket['claim_type']}
Priority: {ticket['priority']}
Policyholder: {customer['name']} <{customer['email']}>
Company: {customer.get('company') or 'N/A'}
Subject: {ticket['subject']}

Description:
{ticket['body']}

CONTEXT
{_format_context(context)}

Write the draft recommendation now. Use the tools available to you to check the
policyholder's plan tier and open-claim load before writing."""


# ------------------------------------------------------------------- agent
def _run_agent(system_prompt: str, user_prompt: str) -> tuple[str, list[dict]]:
    """Run the tool-calling agent. Returns (final_text, tool_calls)."""
    from langchain.agents import create_agent

    agent = create_agent(
        model=get_llm(),
        tools=SUPPORT_TOOLS,
        system_prompt=system_prompt,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    messages = result.get("messages", [])

    tool_calls: list[dict] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls.append({"tool": call.get("name"), "args": call.get("args")})

    final_text = ""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            final_text = content.strip()
            break

    return final_text, tool_calls


def _fallback_generate(system_prompt: str, user_prompt: str) -> str:
    """Plain LLM call with no tools, used when the agent returns nothing."""
    response = get_llm().invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return (response.content or "").strip()


# ------------------------------------------------------------------ public
def generate_recommendation(ticket: dict, customer: dict) -> tuple[str, dict[str, Any]]:
    """Main entry point: claim in, (draft, context_used) out."""
    settings = get_settings()
    context = _gather_context(ticket, customer)
    user_prompt = _build_user_prompt(ticket, customer, context)

    draft = ""
    try:
        draft, tool_calls = _run_agent(SYSTEM_PROMPT, user_prompt)
        context["tool_calls"] = tool_calls
    except Exception as exc:
        logger.exception("agent run failed")
        context["errors"].append(f"agent: {exc}")

    if not draft:
        try:
            draft = _fallback_generate(SYSTEM_PROMPT, user_prompt)
            context["errors"].append("used_fallback_generation")
        except Exception as exc:
            logger.exception("fallback generation failed")
            context["errors"].append(f"llm: {exc}")
            draft = (
                "**Summary**\nAutomatic draft generation is unavailable.\n\n"
                "**Recommended next steps**\n1. Handle this claim manually.\n\n"
                "**Information to request**\nNone\n\n"
                "**Risk / compliance notes**\n- AI draft failed; no AI context applied."
            )

    context["model"] = settings.openai_model
    context["semantic_memory"] = mem.semantic_available()
    return draft, context
```

### 7.3 What just happened — read the flow once more

1. `_gather_context` runs memory + RAG retrieval, **each wrapped in its own try/except** so one failure doesn't kill the other.
2. `_format_context` turns raw hits into labelled text the LLM can read.
3. `_build_user_prompt` puts claim facts and context into one message.
4. `_run_agent` gives the LLM the tools and lets it decide when to call them. LangGraph loops model → tool → model automatically until the model stops calling tools.
5. Every tool call is captured into `context_used` — this is your **transparency trail** for the dashboard.
6. If the agent returns an empty string (it happens with some models and tool loops), `_fallback_generate` runs a plain LLM call.

### ✅ Checkpoint 7

```bash
python - <<'PY'
from customer_support_agent.services.copilot_service import generate_recommendation
from customer_support_agent.repositories import customer_repo, ticket_repo

ticket = ticket_repo.get_ticket(1)
customer = customer_repo.get_by_id(ticket["customer_id"])
draft, ctx = generate_recommendation(ticket, customer)
print(draft)
print("\n--- context ---")
print("memory:", len(ctx["memory_hits"]), "| knowledge:", len(ctx["knowledge_hits"]))
print("tools :", ctx["tool_calls"])
print("errors:", ctx["errors"])
PY
```

Expected: a four-section draft, at least one tool call, and an empty `errors` list.

---

## Phase 8 — Draft lifecycle + human-in-the-loop

**Goal:** generate, edit, approve, discard — and write approved text back to memory.

### 8.1 Draft service

**`customer_support_agent/services/draft_service.py`**

```python
from __future__ import annotations

import logging
from typing import Any, Optional

from customer_support_agent.integrations.memory import langmem_store as mem
from customer_support_agent.repositories import customer_repo, draft_repo, ticket_repo
from customer_support_agent.services.copilot_service import generate_recommendation

logger = logging.getLogger(__name__)


def generate_draft_for_ticket(ticket_id: int) -> Optional[dict[str, Any]]:
    """Generate and persist a draft. Safe to call from a BackgroundTask."""
    try:
        ticket = ticket_repo.get_ticket(ticket_id)
        if not ticket:
            logger.warning("generate_draft: ticket %s not found", ticket_id)
            return None
        customer = customer_repo.get_by_id(ticket["customer_id"])
        content, context = generate_recommendation(ticket, customer)
        return draft_repo.create_draft(ticket_id, content, context)
    except Exception:
        logger.exception("Background draft generation failed for ticket %s", ticket_id)
        return None


def accept_draft(
    draft_id: int, final_content: Optional[str], save_to_memory: bool
) -> dict[str, Any]:
    """Approve a draft; optionally persist the resolution into memory."""
    draft = draft_repo.get_draft(draft_id)
    if not draft:
        raise ValueError("Draft not found")

    updated = draft_repo.update_draft(
        draft_id, content=final_content, status="accepted"
    )
    ticket = ticket_repo.get_ticket(updated["ticket_id"])
    ticket_repo.set_status(ticket["id"], "resolved")
    customer = customer_repo.get_by_id(ticket["customer_id"])

    memory_result: dict[str, Any] = {"saved": False, "reason": "not requested"}
    if save_to_memory:
        try:
            memory_result = mem.save_memory(
                content=(
                    f"Claim: {ticket['subject']} ({ticket['claim_type']}). "
                    f"Approved resolution: {updated['content']}"
                ),
                email=customer["email"],
                company=customer.get("company"),
                metadata={
                    "ticket_id": ticket["id"],
                    "claim_type": ticket["claim_type"],
                    "priority": ticket["priority"],
                },
            )
        except Exception as exc:
            logger.exception("memory write failed")
            memory_result = {"saved": False, "reason": str(exc)}

    return {"draft": updated, "memory": memory_result}


def discard_draft(draft_id: int) -> dict[str, Any]:
    return draft_repo.update_draft(draft_id, content=None, status="discarded")
```

### 8.2 Drafts router

Replace **`customer_support_agent/api/routers/drafts.py`**:

```python
from fastapi import APIRouter, HTTPException

from customer_support_agent.models.schemas import AcceptDraft, DraftOut, DraftUpdate
from customer_support_agent.repositories import draft_repo
from customer_support_agent.services import draft_service

router = APIRouter(prefix="/drafts", tags=["drafts"])


@router.post("/generate/{ticket_id}", response_model=DraftOut)
def generate(ticket_id: int):
    draft = draft_service.generate_draft_for_ticket(ticket_id)
    if not draft:
        raise HTTPException(500, "Draft generation failed")
    return draft


@router.get("/ticket/{ticket_id}", response_model=DraftOut)
def latest(ticket_id: int):
    draft = draft_repo.latest_for_ticket(ticket_id)
    if not draft:
        raise HTTPException(404, "No draft for this ticket yet")
    return draft


@router.patch("/{draft_id}", response_model=DraftOut)
def edit(draft_id: int, payload: DraftUpdate):
    if not draft_repo.get_draft(draft_id):
        raise HTTPException(404, "Draft not found")
    return draft_repo.update_draft(draft_id, content=payload.content, status=None)


@router.post("/{draft_id}/accept")
def accept(draft_id: int, payload: AcceptDraft):
    try:
        return draft_service.accept_draft(
            draft_id, payload.content, payload.save_to_memory
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.post("/{draft_id}/discard", response_model=DraftOut)
def discard(draft_id: int):
    if not draft_repo.get_draft(draft_id):
        raise HTTPException(404, "Draft not found")
    return draft_service.discard_draft(draft_id)
```

### 8.3 Wire background generation into ticket creation

In `customer_support_agent/api/routers/tickets.py`, uncomment the background task:

```python
from customer_support_agent.services.draft_service import generate_draft_for_ticket
...
    background.add_task(generate_draft_for_ticket, ticket["id"])
```

### 8.4 Add a memory-probe endpoint (used by the dashboard)

Create **`customer_support_agent/api/routers/memory.py`**:

```python
from fastapi import APIRouter, Query

from customer_support_agent.integrations.memory import langmem_store as mem

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/search")
def search(
    q: str = Query(..., min_length=3),
    email: str = Query(...),
    company: str | None = None,
):
    return {
        "query": q,
        "semantic": mem.semantic_available(),
        "hits": mem.search_memories(q, email=email, company=company),
    }
```

Register it in `app_factory.py`:

```python
from customer_support_agent.api.routers import memory
app.include_router(memory.router)
```

### ✅ Checkpoint 8

Restart the server, then:

```bash
# 1. Create a claim (draft generates in the background)
curl -X POST http://localhost:8000/tickets -H "Content-Type: application/json" -d '{
  "customer": {"name":"Ravi Kumar","email":"ravi@acme.com","company":"Acme Ltd","plan_tier":"premium"},
  "subject":"Windshield cracked by road debris",
  "body":"A stone hit my windshield on the highway. The car is drivable. No injuries.",
  "claim_type":"auto","priority":"medium"}'

# 2. Wait ~10 seconds, then read the draft
curl http://localhost:8000/drafts/ticket/2

# 3. Approve it and write to memory
curl -X POST http://localhost:8000/drafts/1/accept \
  -H "Content-Type: application/json" \
  -d '{"save_to_memory": true}'

# 4. Prove the loop closed
curl "http://localhost:8000/memory/search?q=windshield%20damage&email=ravi@acme.com&company=Acme%20Ltd"
```

Expected: step 4 returns the resolution you just approved. **The memory loop is now closed — this is the core of the project.**

---

## Phase 9 — Streamlit dashboard

**Goal:** the operator UI from the architecture diagram.

**`app.py`** (project root)

```python
import os

import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 120

st.set_page_config(page_title="Claims Copilot", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------ helpers
def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict | None = None):
    r = requests.post(f"{API}{path}", json=payload or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_patch(path: str, payload: dict):
    r = requests.patch(f"{API}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("System")
    try:
        health = api_get("/health")
        st.success("API online")
        st.caption(f"Model: {health.get('model', 'n/a')}")
        st.caption(f"LLM configured: {health['llm_configured']}")
        st.caption(f"Semantic enabled: {health['semantic_enabled']}")
    except Exception as exc:
        st.error(f"API unreachable: {exc}")
        st.stop()

    st.divider()
    st.header("Knowledge base")
    if st.button("Ingest knowledge base", use_container_width=True):
        with st.spinner("Chunking and indexing..."):
            st.json(api_post("/knowledge/ingest"))
    try:
        st.caption(f"Indexed chunks: {api_get('/knowledge/stats')['chunks']}")
    except Exception:
        pass

st.title("🛡️ Insurance Claims Support Copilot")
tab_new, tab_review, tab_memory = st.tabs(
    ["Register claim", "Review drafts", "Claim history"]
)


# ------------------------------------------------------- tab 1: register FNOL
with tab_new:
    st.subheader("First Notice of Loss")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Policyholder name")
        email = st.text_input("Email")
        company = st.text_input("Company", value="")
    with col2:
        plan = st.selectbox("Plan tier", ["basic", "standard", "premium"], index=1)
        claim_type = st.selectbox("Claim type", ["auto", "property", "health", "other"])
        priority = st.selectbox("Priority", ["low", "medium", "high"], index=1)

    subject = st.text_input("Subject")
    body = st.text_area("Incident description", height=160)

    if st.button("Register claim", type="primary"):
        if not (name and email and subject and len(body) >= 10):
            st.warning("Fill in name, email, subject, and a description.")
        else:
            ticket = api_post(
                "/tickets",
                {
                    "customer": {
                        "name": name, "email": email,
                        "company": company or None, "plan_tier": plan,
                    },
                    "subject": subject, "body": body,
                    "claim_type": claim_type, "priority": priority,
                },
            )
            st.success(f"Claim #{ticket['id']} registered. Draft generating...")
            st.session_state["selected_ticket"] = ticket["id"]


# --------------------------------------------------------- tab 2: review loop
with tab_review:
    tickets = api_get("/tickets", limit=50)
    if not tickets:
        st.info("No claims yet. Register one in the first tab.")
    else:
        labels = {
            f"#{t['id']} · {t['subject'][:48]} · {t['status']}": t["id"]
            for t in tickets
        }
        chosen = st.selectbox("Select a claim", list(labels))
        ticket_id = labels[chosen]
        ticket = api_get(f"/tickets/{ticket_id}")

        with st.expander("Claim details", expanded=False):
            st.write(ticket["body"])
            st.caption(
                f"Type: {ticket['claim_type']} · Priority: {ticket['priority']} · "
                f"Status: {ticket['status']}"
            )

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Generate / regenerate", use_container_width=True):
                with st.spinner("Copilot is drafting..."):
                    api_post(f"/drafts/generate/{ticket_id}")
                st.rerun()

        try:
            draft = api_get(f"/drafts/ticket/{ticket_id}")
        except requests.HTTPError:
            st.info("No draft yet — click Generate.")
            draft = None

        if draft:
            st.caption(f"Draft #{draft['id']} · status: {draft['status']}")
            edited = st.text_area(
                "AI recommendation (editable)", value=draft["content"], height=340
            )
            save_mem = st.checkbox("Save approved resolution to memory", value=True)

            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Approve", type="primary", use_container_width=True):
                result = api_post(
                    f"/drafts/{draft['id']}/accept",
                    {"content": edited, "save_to_memory": save_mem},
                )
                st.success("Approved.")
                st.json(result["memory"])
            if b2.button("💾 Save edit", use_container_width=True):
                api_patch(f"/drafts/{draft['id']}", {"content": edited})
                st.success("Saved.")
            if b3.button("🗑️ Discard", use_container_width=True):
                api_post(f"/drafts/{draft['id']}/discard")
                st.warning("Discarded.")
                st.rerun()

            # --- transparency panel ---
            ctx = draft.get("context_used") or {}
            st.divider()
            st.subheader("Context used by the copilot")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Memory hits", len(ctx.get("memory_hits", [])))
            m2.metric("Knowledge hits", len(ctx.get("knowledge_hits", [])))
            m3.metric("Tool calls", len(ctx.get("tool_calls", [])))
            m4.metric("Errors", len(ctx.get("errors", [])))

            with st.expander("Memory hits"):
                for h in ctx.get("memory_hits", []):
                    st.markdown(
                        f"- `{h['metadata']['scope']}` (score {h['score']}) — {h['memory']}"
                    )
            with st.expander("Knowledge hits"):
                for h in ctx.get("knowledge_hits", []):
                    st.markdown(f"**{h['source']}** (score {h['score']})")
                    st.caption(h["text"][:400])
            with st.expander("Tool calls"):
                st.json(ctx.get("tool_calls", []))
            if ctx.get("errors"):
                st.error(ctx["errors"])


# --------------------------------------------------------- tab 3: memory probe
with tab_memory:
    st.subheader("Probe claim-resolution memory")
    q = st.text_input("Query", "deductible waiver for rear-end collision")
    e = st.text_input("Policyholder email", "ravi@acme.com")
    co = st.text_input("Company", "Acme Ltd")
    if st.button("Search memory"):
        result = api_get("/memory/search", q=q, email=e, company=co)
        st.caption(f"Semantic search enabled: {result['semantic']}")
        if not result["hits"]:
            st.info("No memories found for this scope yet.")
        for h in result["hits"]:
            st.markdown(f"**{h['metadata']['scope']}** · score {h['score']}")
            st.write(h["memory"])
            st.divider()
```

### ✅ Checkpoint 9

Two terminals:

```bash
# Terminal 1
uvicorn customer_support_agent.main:app --reload --port 8000
# Terminal 2
streamlit run app.py
```

Walk the full loop in the browser: ingest KB → register a claim → review the draft → check the context panel → approve → probe memory. **This is your demo.** Record it.

---

## Phase 10 — Testing with pytest

**Goal:** tests that run in CI with no API keys.

**`tests/conftest.py`**

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def temp_env(tmp_path_factory):
    """Point the app at a throwaway database and vector store."""
    import os

    tmp = tmp_path_factory.mktemp("copilot")
    os.environ["DB_PATH"] = str(tmp / "test.db")
    os.environ["CHROMA_PATH"] = str(tmp / "chroma")
    os.environ["ENABLE_SEMANTIC"] = "false"      # no network calls in CI
    os.environ.setdefault("OPENAI_API_KEY", "")
    from customer_support_agent.core.settings import get_settings

    get_settings.cache_clear()
    yield


@pytest.fixture
def client(temp_env):
    from customer_support_agent.main import app

    with TestClient(app) as c:
        yield c
```

**`tests/test_api.py`**

```python
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_list_ticket(client, monkeypatch):
    # Stop the background task from calling a real LLM.
    monkeypatch.setattr(
        "customer_support_agent.services.draft_service.generate_draft_for_ticket",
        lambda ticket_id: None,
    )
    payload = {
        "customer": {"name": "Test User", "email": "T@Example.com",
                     "company": "Acme", "plan_tier": "premium"},
        "subject": "Bumper damage",
        "body": "Minor collision in a parking lot, no injuries reported.",
        "claim_type": "auto", "priority": "low",
    }
    r = client.post("/tickets", json=payload)
    assert r.status_code == 201
    ticket_id = r.json()["id"]

    assert client.get(f"/tickets/{ticket_id}").status_code == 200
    assert any(t["id"] == ticket_id for t in client.get("/tickets").json())
```

**`tests/test_memory.py`**

```python
from customer_support_agent.integrations.memory import langmem_store as mem


def test_namespaces_are_normalized():
    assert mem.customer_namespace("  Asha.Rao@Example.COM ")[-1] == "asha.rao@example.com"
    assert mem.company_namespace("Acme Ltd.")[-1] == "company::acme-ltd"
    assert mem.company_namespace(None)[-1] == "company::general"


def test_save_and_search_roundtrip():
    mem.save_memory("Deductible waived under the no-fault clause.",
                    email="a@b.com", company="Acme")
    hits = mem.search_memories("deductible", email="a@b.com", company="Acme")
    assert hits
    assert "deductible" in hits[0]["memory"].lower()


def test_search_is_deduplicated():
    mem.save_memory("Identical resolution text.", email="c@d.com", company="Zeta")
    hits = mem.search_memories("identical", email="c@d.com", company="Zeta")
    texts = [h["memory"] for h in hits]
    assert len(texts) == len(set(texts))
```

**`tests/test_copilot.py`**

```python
from customer_support_agent.services import copilot_service


def test_context_formatting_handles_empty_hits():
    empty = {"memory_hits": [], "knowledge_hits": [], "tool_calls": [], "errors": []}
    text = copilot_service._format_context(empty)
    assert "none found" in text.lower()


def test_generate_recommendation_falls_back(monkeypatch):
    monkeypatch.setattr(
        copilot_service, "_gather_context",
        lambda t, c: {"memory_hits": [], "knowledge_hits": [],
                      "tool_calls": [], "errors": []},
    )
    monkeypatch.setattr(
        copilot_service, "_run_agent",
        lambda s, u: (_ for _ in ()).throw(RuntimeError("agent down")),
    )
    monkeypatch.setattr(
        copilot_service, "_fallback_generate", lambda s, u: "**Summary**\nFallback."
    )
    ticket = {"id": 1, "subject": "s", "body": "b", "claim_type": "auto",
              "priority": "low"}
    customer = {"name": "n", "email": "e@f.com", "company": "Acme"}
    draft, ctx = copilot_service.generate_recommendation(ticket, customer)
    assert "Fallback" in draft
    assert "used_fallback_generation" in ctx["errors"]
```

### ✅ Checkpoint 10

```bash
uv run pytest -v
```

All tests pass with **no API keys set**. That is the requirement for CI.

---

## Phase 11 — Docker + Docker Compose

**Goal:** one command starts both services.

**`Dockerfile`**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -r pyproject.toml

COPY . .

RUN mkdir -p /app/data
EXPOSE 8000 8501

CMD ["uvicorn", "customer_support_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`.dockerignore`**

```
.venv/
__pycache__/
data/
.git/
.pytest_cache/
*.md
.env
```

**`docker-compose.yml`**

```yaml
services:
  api:
    build: .
    container_name: claims-api
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./knowledge_base:/app/knowledge_base:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    restart: unless-stopped

  dashboard:
    build: .
    container_name: claims-dashboard
    env_file: .env
    environment:
      API_BASE_URL: http://api:8000
    command: >
      streamlit run app.py
      --server.port=8501
      --server.address=0.0.0.0
      --server.headless=true
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
```

> **`API_BASE_URL: http://api:8000`** — inside Compose, containers reach each other by **service name**, never by `localhost`. This is the mistake beginners hit here.

### ✅ Checkpoint 11

```bash
docker compose up --build
```

Visit `http://localhost:8501` and run the full loop again. Then `docker compose down` and `docker compose up` — the SQLite data survives (volume-mounted), the LangMem data does not (in-process RAM). Know why, and be ready to say so.

---

## Phase 12 — CI/CD + EC2 deployment

### 12.1 GitHub Actions

**`.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Lint
        run: uv run ruff check .

      - name: Run tests
        env:
          OPENAI_API_KEY: ""
          ENABLE_SEMANTIC: "false"
        run: uv run pytest -v

  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t claims-copilot:${{ github.sha }} .
```

### 12.2 EC2 deployment

**Launch:** Ubuntu 22.04, `t3.small` or larger (ChromaDB plus two Python processes needs ≥2 GB RAM). Security group inbound: `22` from your IP only, `8501` from your IP, `8000` only if you want the API public.

**Install Docker on the instance:**

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker ubuntu
newgrp docker
```

**Deploy:**

```bash
git clone https://github.com/<you>/insurance-claims-copilot.git
cd insurance-claims-copilot

cp .env.example .env
nano .env                # paste the real OPENAI_API_KEY

docker compose up -d --build
docker compose ps
docker compose logs -f api
```

Open `http://<EC2_PUBLIC_IP>:8501`, click **Ingest knowledge base** once, and run the loop.

**Redeploy after a code change:**

```bash
cd insurance-claims-copilot && git pull && docker compose up -d --build
```

### ✅ Checkpoint 12

CI is green on GitHub, and the dashboard is reachable on the EC2 public IP.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | `.env` not loaded | Run commands from the project root; check `env_file=".env"` in settings |
| Draft is empty, `used_fallback_generation` in errors | Agent ended on a tool message | Expected — the fallback handles it. Check tool docstrings if it happens every time |
| `chromadb` import error on Apple Silicon | Missing build tools | `pip install --upgrade chromadb` inside the venv; or use Docker |
| Memory search always returns score `0.0` | `ENABLE_SEMANTIC=false`, or embeddings failed | Expected fallback behaviour. Check the log for an embeddings warning |
| Memory empty after restart | `InMemoryStore` is RAM-only | Expected. Swap for a Postgres store to persist |
| Streamlit shows "API unreachable" in Docker | Used `localhost` instead of service name | Set `API_BASE_URL=http://api:8000` |
| Ingest returns 0 chunks | Empty `knowledge_base/` | Add `.md` files; check the path in `/knowledge/stats` |
| `429 rate_limit_exceeded` | OpenAI tier limits | `max_retries` already backs off. Lower `rag_top_k`, shorten prompts, or raise your usage tier |
| `401 invalid_api_key` | Key typo, or key from a different org | Regenerate at platform.openai.com/api-keys; keys start with `sk-` |
| `insufficient_quota` | No billing on the OpenAI account | Add a payment method — the free trial does not cover the API |
| `Embedding dimension mismatch` in Chroma | Changed `EMBEDDING_MODEL` or `EMBEDDING_DIMS` | Delete `data/chroma/`, re-ingest, restart the API |
| `temperature` rejected by the API | Using a reasoning model | Remove `temperature` from `ChatOpenAI` |
| `TypeError` on `create_agent` import | Older LangChain | `uv add "langchain>=1.0"` — pre-1.0 uses `create_react_agent` from `langgraph.prebuilt` |

---

## Suggested build schedule

| Day | Phases | Outcome at end of day |
|---|---|---|
| 1 | 0–3 | API runs, tickets save to SQLite |
| 2 | 4–6 | RAG search and tools both return real results |
| 3 | 7–8 | Full draft → approve → memory loop works via curl |
| 4 | 9–10 | Dashboard demo + green test suite |
| 5 | 11–12 | Dockerized, CI green, live on EC2 |

---

## What to highlight when you present this project

1. **Three distinct context sources, deliberately separated** — memory (experience), RAG (documents), tools (live facts). Explain why merging them into one retriever is worse.
2. **Graceful degradation at every layer** — no API key, no embeddings, no memory hits, empty agent output. The copilot always produces something, and the failure is recorded in `context_used`.
3. **Human-in-the-loop by design, not as an afterthought** — the AI never sets claim status by itself, and the human decides whether the resolution enters memory. In a regulated domain that is a product requirement, not a nicety.
4. **The learning loop** — approved resolutions become retrievable memory, so the fifth similar claim is handled better than the first.
5. **Transparency** — every draft carries the exact memory hits, knowledge chunks, and tool calls that produced it. That is what makes an AI decision auditable.

---

## Extensions once the base works

- Swap `InMemoryStore` for `PostgresStore` so memory survives restarts
- Add a `check_fraud_indicators` tool that scores the claim text against the fraud-indicator knowledge file
- Add LangSmith tracing (`LANGCHAIN_TRACING_V2=true`) to see every agent step
- Add a reranker over the Chroma top-k before it hits the prompt
- Add per-draft evaluation: groundedness, and whether the human edited it (an implicit quality signal)
- Add auth (API key header or JWT) before exposing the API publicly