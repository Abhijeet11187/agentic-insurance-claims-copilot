from fastapi import APIRouter, BackgroundTasks, HTTPException

from customer_support_agent.models.schemas import TicketIn, TicketOut
from customer_support_agent.repositories import customer_repo, ticket_repo
from customer_support_agent.services.draft_service import generate_draft_for_ticket

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

    # Phase 8 wires the copilot in here.

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