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