import os

import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 120

st.set_page_config(page_title="Claims Copilot", page_icon="🛡️", layout="wide")


def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict | None = None):
    r = requests.post(f"{API}{path}", json=payload or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_patch(path: str, payload: dict):
    r = requests.patch(f"{API}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


with st.sidebar:
    st.header("System")
    try:
        health = api_get("/health")
        st.success("API online")
        st.caption(f"Model: {health.get('model', 'n/a')}")
        st.caption(f"LLM configured: {health['llm_configured']}")
        st.caption(f"Semantic enabled: {health['semantic_enabled']}")
    except Exception as exc:
        st.error(f"API unreachable: {exc}")
        st.stop()

    st.divider()
    st.header("Knowledge base")
    if st.button("Ingest knowledge base", use_container_width=True):
        with st.spinner("Chunking and indexing..."):
            st.json(api_post("/knowledge/ingest"))
    try:
        st.caption(f"Indexed chunks: {api_get('/knowledge/stats')['chunks']}")
    except Exception:
        pass

st.title("🛡️ Insurance Claims Support Copilot")
tab_new, tab_review, tab_memory = st.tabs(
    ["Register claim", "Review drafts", "Claim history"]
)


with tab_new:
    st.subheader("First Notice of Loss")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Policyholder name")
        email = st.text_input("Email")
        company = st.text_input("Company", value="")
    with col2:
        plan = st.selectbox("Plan tier", ["basic", "standard", "premium"], index=1)
        claim_type = st.selectbox("Claim type", ["auto", "property", "health", "other"])
        priority = st.selectbox("Priority", ["low", "medium", "high"], index=1)

    subject = st.text_input("Subject")
    body = st.text_area("Incident description", height=160)

    if st.button("Register claim", type="primary"):
        if not (name and email and subject and len(body) >= 10):
            st.warning("Fill in name, email, subject, and a description.")
        else:
            ticket = api_post(
                "/tickets",
                {
                    "customer": {
                        "name": name, "email": email,
                        "company": company or None, "plan_tier": plan,
                    },
                    "subject": subject, "body": body,
                    "claim_type": claim_type, "priority": priority,
                },
            )
            st.success(f"Claim #{ticket['id']} registered. Draft generating...")


with tab_review:
    tickets = api_get("/tickets", limit=50)
    if not tickets:
        st.info("No claims yet. Register one in the first tab.")
    else:
        labels = {
            f"#{t['id']} · {t['subject'][:48]} · {t['status']}": t["id"]
            for t in tickets
        }
        chosen = st.selectbox("Select a claim", list(labels))
        ticket_id = labels[chosen]
        ticket = api_get(f"/tickets/{ticket_id}")

        with st.expander("Claim details", expanded=False):
            st.write(ticket["body"])
            st.caption(
                f"Type: {ticket['claim_type']} · Priority: {ticket['priority']} · "
                f"Status: {ticket['status']}"
            )

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Generate / regenerate", use_container_width=True):
                with st.spinner("Copilot is drafting..."):
                    api_post(f"/drafts/generate/{ticket_id}")
                st.rerun()

        try:
            draft = api_get(f"/drafts/ticket/{ticket_id}")
        except requests.HTTPError:
            st.info("No draft yet — click Generate.")
            draft = None

        if draft:
            st.caption(f"Draft #{draft['id']} · status: {draft['status']}")
            edited = st.text_area(
                "AI recommendation (editable)", value=draft["content"], height=340
            )
            save_mem = st.checkbox("Save approved resolution to memory", value=True)

            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Approve", type="primary", use_container_width=True):
                result = api_post(
                    f"/drafts/{draft['id']}/accept",
                    {"content": edited, "save_to_memory": save_mem},
                )
                st.success("Approved.")
                st.json(result["memory"])
            if b2.button("💾 Save edit", use_container_width=True):
                api_patch(f"/drafts/{draft['id']}", {"content": edited})
                st.success("Saved.")
            if b3.button("🗑️ Discard", use_container_width=True):
                api_post(f"/drafts/{draft['id']}/discard")
                st.warning("Discarded.")
                st.rerun()

            ctx = draft.get("context_used") or {}
            st.divider()
            st.subheader("Context used by the copilot")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Memory hits", len(ctx.get("memory_hits", [])))
            m2.metric("Knowledge hits", len(ctx.get("knowledge_hits", [])))
            m3.metric("Tool calls", len(ctx.get("tool_calls", [])))
            m4.metric("Errors", len(ctx.get("errors", [])))

            with st.expander("Memory hits"):
                for h in ctx.get("memory_hits", []):
                    st.markdown(
                        f"- `{h['metadata']['scope']}` (score {h['score']}) — {h['memory']}"
                    )
            with st.expander("Knowledge hits"):
                for h in ctx.get("knowledge_hits", []):
                    st.markdown(f"**{h['source']}** (score {h['score']})")
                    st.caption(h["text"][:400])
            with st.expander("Tool calls"):
                st.json(ctx.get("tool_calls", []))
            if ctx.get("errors"):
                st.error(ctx["errors"])


with tab_memory:
    st.subheader("Probe claim-resolution memory")
    q = st.text_input("Query", "deductible waiver for rear-end collision")
    e = st.text_input("Policyholder email", "ravi@acme.com")
    co = st.text_input("Company", "Acme Ltd")
    if st.button("Search memory"):
        result = api_get("/memory/search", q=q, email=e, company=co)
        st.caption(f"Semantic search enabled: {result['semantic']}")
        if not result["hits"]:
            st.info("No memories found for this scope yet.")
        for h in result["hits"]:
            st.markdown(f"**{h['metadata']['scope']}** · score {h['score']}")
            st.write(h["memory"])
            st.divider()