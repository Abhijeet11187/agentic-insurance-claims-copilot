from __future__ import annotations

import logging
from typing import Any

from customer_support_agent.core.settings import get_settings
from customer_support_agent.integrations.llm.openai_client import get_llm
from customer_support_agent.integrations.memory import langmem_store as mem
from customer_support_agent.integrations.rag import chroma_kb
from customer_support_agent.integrations.tools.support_tools import SUPPORT_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an internal AI copilot for insurance claims support agents.
You do NOT talk to the customer and you do NOT make final claim decisions.
You write a draft recommendation that a licensed human adjuster will review and edit.

Rules you must follow:
1. Ground every factual statement in the CONTEXT provided. If the context does not
   cover something, write "Needs verification: <what is missing>" instead of guessing.
2. Never promise, estimate, or imply a settlement amount.
3. Never state a final liability or coverage decision. Recommend, do not decide.
4. Use the operational tools available to you when plan tier, SLA, or the
   policyholder's open-claim load is relevant.
5. Cite the knowledge source filename in brackets when you rely on a policy document.

Output exactly these four sections in markdown:

**Summary** - two sentences on what happened.
**Recommended next steps** - 3 to 5 numbered actions for the adjuster.
**Information to request** - bullets, or "None" if nothing is missing.
**Risk / compliance notes** - bullets covering fraud indicators, SLA risk, or
escalation triggers. Write "None identified" if there are none.

Keep the whole draft under 300 words."""


# ------------------------------------------------------------------ context
def _gather_context(ticket: dict, customer: dict) -> dict[str, Any]:
    query = f"{ticket['subject']} {ticket['body']}"
    context: dict[str, Any] = {
        "memory_hits": [], "knowledge_hits": [], "tool_calls": [], "errors": [],
    }

    try:
        context["memory_hits"] = mem.search_memories(
            query, email=customer["email"], company=customer.get("company")
        )
    except Exception as exc:
        logger.exception("memory retrieval failed")
        context["errors"].append(f"memory: {exc}")

    try:
        context["knowledge_hits"] = chroma_kb.search_knowledge(query)
    except Exception as exc:
        logger.exception("knowledge retrieval failed")
        context["errors"].append(f"knowledge: {exc}")

    return context


def _format_context(context: dict[str, Any]) -> str:
    parts: list[str] = []

    if context["memory_hits"]:
        lines = [
            f"- [{h['metadata']['scope']} scope] {h['memory']}"
            for h in context["memory_hits"]
        ]
        parts.append("PAST RESOLUTIONS (memory):\n" + "\n".join(lines))
    else:
        parts.append("PAST RESOLUTIONS (memory): none found.")

    if context["knowledge_hits"]:
        lines = [f"- [{h['source']}] {h['text']}" for h in context["knowledge_hits"]]
        parts.append("POLICY KNOWLEDGE (retrieved):\n" + "\n".join(lines))
    else:
        parts.append("POLICY KNOWLEDGE (retrieved): none found.")

    return "\n\n".join(parts)


def _build_user_prompt(ticket: dict, customer: dict, context: dict) -> str:
    return f"""CLAIM DETAILS
Ticket ID: {ticket['id']}
Claim type: {ticket['claim_type']}
Priority: {ticket['priority']}
Policyholder: {customer['name']} <{customer['email']}>
Company: {customer.get('company') or 'N/A'}
Subject: {ticket['subject']}

Description:
{ticket['body']}

CONTEXT
{_format_context(context)}

Write the draft recommendation now. Use the tools available to you to check the
policyholder's plan tier and open-claim load before writing."""


# ------------------------------------------------------------------- agent
def _run_agent(system_prompt: str, user_prompt: str) -> tuple[str, list[dict]]:
    """Run the tool-calling agent. Returns (final_text, tool_calls)."""
    from langchain.agents import create_agent

    agent = create_agent(
        model=get_llm(),
        tools=SUPPORT_TOOLS,
        system_prompt=system_prompt,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    messages = result.get("messages", [])

    tool_calls: list[dict] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls.append({"tool": call.get("name"), "args": call.get("args")})

    final_text = ""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            final_text = content.strip()
            break

    return final_text, tool_calls


def _fallback_generate(system_prompt: str, user_prompt: str) -> str:
    """Plain LLM call with no tools, used when the agent returns nothing."""
    response = get_llm().invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return (response.content or "").strip()


# ------------------------------------------------------------------ public
def generate_recommendation(ticket: dict, customer: dict) -> tuple[str, dict[str, Any]]:
    """Main entry point: claim in, (draft, context_used) out."""
    settings = get_settings()
    context = _gather_context(ticket, customer)
    user_prompt = _build_user_prompt(ticket, customer, context)

    draft = ""
    try:
        draft, tool_calls = _run_agent(SYSTEM_PROMPT, user_prompt)
        context["tool_calls"] = tool_calls
    except Exception as exc:
        logger.exception("agent run failed")
        context["errors"].append(f"agent: {exc}")

    if not draft:
        try:
            draft = _fallback_generate(SYSTEM_PROMPT, user_prompt)
            context["errors"].append("used_fallback_generation")
        except Exception as exc:
            logger.exception("fallback generation failed")
            context["errors"].append(f"llm: {exc}")
            draft = (
                "**Summary**\nAutomatic draft generation is unavailable.\n\n"
                "**Recommended next steps**\n1. Handle this claim manually.\n\n"
                "**Information to request**\nNone\n\n"
                "**Risk / compliance notes**\n- AI draft failed; no AI context applied."
            )

    context["model"] = settings.openai_model
    context["semantic_memory"] = mem.semantic_available()
    return draft, context