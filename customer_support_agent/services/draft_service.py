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