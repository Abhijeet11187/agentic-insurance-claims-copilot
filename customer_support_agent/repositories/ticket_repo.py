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