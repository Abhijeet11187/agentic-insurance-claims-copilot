from fastapi import APIRouter, HTTPException

from customer_support_agent.models.schemas import AcceptDraft, DraftOut, DraftUpdate
from customer_support_agent.repositories import draft_repo
from customer_support_agent.services import draft_service

router = APIRouter(prefix="/drafts", tags=["drafts"])


@router.post("/generate/{ticket_id}", response_model=DraftOut)
def generate(ticket_id: int):
    draft = draft_service.generate_draft_for_ticket(ticket_id)
    if not draft:
        raise HTTPException(500, "Draft generation failed")
    return draft


@router.get("/ticket/{ticket_id}", response_model=DraftOut)
def latest(ticket_id: int):
    draft = draft_repo.latest_for_ticket(ticket_id)
    if not draft:
        raise HTTPException(404, "No draft for this ticket yet")
    return draft


@router.patch("/{draft_id}", response_model=DraftOut)
def edit(draft_id: int, payload: DraftUpdate):
    if not draft_repo.get_draft(draft_id):
        raise HTTPException(404, "Draft not found")
    return draft_repo.update_draft(draft_id, content=payload.content, status=None)


@router.post("/{draft_id}/accept")
def accept(draft_id: int, payload: AcceptDraft):
    try:
        return draft_service.accept_draft(
            draft_id, payload.content, payload.save_to_memory
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@router.post("/{draft_id}/discard", response_model=DraftOut)
def discard(draft_id: int):
    if not draft_repo.get_draft(draft_id):
        raise HTTPException(404, "Draft not found")
    return draft_service.discard_draft(draft_id)