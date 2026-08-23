from customer_support_agent.repositories import customer_repo, ticket_repo
from customer_support_agent.services.copilot_service import generate_recommendation

ticket = ticket_repo.get_ticket(1)
customer = customer_repo.get_by_id(ticket["customer_id"])

draft, ctx = generate_recommendation(ticket, customer)

print(draft)
print("\n" + "=" * 60)
print("memory hits   :", len(ctx["memory_hits"]))
print("knowledge hits:", len(ctx["knowledge_hits"]))
print("tool calls    :", ctx["tool_calls"])
print("errors        :", ctx["errors"])