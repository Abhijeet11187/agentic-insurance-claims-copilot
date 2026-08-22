from customer_support_agent.integrations.memory import langmem_store as mem

print("semantic:", mem.semantic_available())

mem.save_memory(
    "Rear-end collision, third party at fault per police report. Deductible waived "
    "under the liability-admitted clause. Repair approved at network garage. "
    "Settled in 6 days.",
    email="Asha.Rao@example.com",
    company="Acme Ltd",
    metadata={"claim_type": "auto", "ticket_id": 1},
)

for hit in mem.search_memories(
    "someone hit me from behind, who pays the excess?",
    email="asha.rao@example.com",
    company="Acme Ltd",
):
    print(f"[{hit['metadata']['scope']}] {hit['score']} | {hit['memory'][:70]}")