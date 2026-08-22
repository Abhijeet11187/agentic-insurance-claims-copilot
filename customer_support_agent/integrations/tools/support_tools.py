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