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