from customer_support_agent.db.database import init_db
from customer_support_agent.repositories import customer_repo, ticket_repo

init_db()

c = customer_repo.upsert_customer(
    "Asha Rao", "Asha.Rao@Example.com", "Acme Ltd", "premium"
)
t = ticket_repo.create_ticket(
    c["id"], "Rear-end collision", "Hit from behind at a signal.", "auto", "high"
)

print("customer:", c)
print("ticket  :", t)
print("open    :", ticket_repo.count_open_for_customer(c["id"]))