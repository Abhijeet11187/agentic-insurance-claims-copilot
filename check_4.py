from customer_support_agent.integrations.tools.support_tools import (
    lookup_customer_plan,
    lookup_open_ticket_load,
)

print(lookup_customer_plan.invoke({"email": "asha.rao@example.com"}))
print(lookup_open_ticket_load.invoke({"email": "asha.rao@example.com"}))
print("---")
print("name:", lookup_customer_plan.name)
print("desc:", lookup_customer_plan.description[:80])