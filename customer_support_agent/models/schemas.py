from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class CustomerIn(BaseModel):
    name: str
    email: EmailStr
    company: Optional[str] = None
    plan_tier: Literal["basic", "standard", "premium"] = "standard"


class CustomerOut(CustomerIn):
    id: int


class TicketIn(BaseModel):
    """A First Notice of Loss (FNOL) / claim intake."""
    customer: CustomerIn
    subject: str = Field(..., min_length=3)
    body: str = Field(..., min_length=10)
    claim_type: Literal["auto", "property", "health", "other"] = "auto"
    priority: Literal["low", "medium", "high"] = "medium"


class TicketOut(BaseModel):
    id: int
    customer_id: int
    subject: str
    body: str
    claim_type: str
    priority: str
    status: str
    created_at: str


class DraftOut(BaseModel):
    id: int
    ticket_id: int
    content: str
    status: str
    context_used: dict[str, Any] = {}
    created_at: str
    updated_at: str


class DraftUpdate(BaseModel):
    content: str


class AcceptDraft(BaseModel):
    content: Optional[str] = None
    save_to_memory: bool = True