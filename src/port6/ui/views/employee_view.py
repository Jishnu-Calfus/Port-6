import streamlit as st

from port6.schemas import Citation
from port6.ui import api_client

ASSISTANT_AVATAR = "📄"


def _dedup_citations(citations: list[Citation]) -> list[Citation]:
    """Multiple reranked chunks often land on the same page — one badge per
    (document, page) is enough; showing the same source twice just adds
    visual noise."""
    seen: set[tuple[str, int]] = set()
    deduped: list[Citation] = []
    for citation in citations:
        key = (citation.source_filename, citation.page_number)
        if key not in seen:
            seen.add(key)
            deduped.append(citation)
    return deduped


def _render_citations(citations: list[Citation]) -> None:
    """A bordered 'Sources' panel instead of a row of popovers — popovers
    render as floating overlays that can visually collide with the chat
    message below them once the answer text runs long. A panel that pushes
    content down instead of overlaying it stays legible regardless of
    answer length."""
    citations = _dedup_citations(citations)
    if not citations:
        return

    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for citation in citations:
            with st.container(border=True):
                st.markdown(f"**{citation.source_filename}** · page {citation.page_number}")
                st.caption(citation.snippet)


def render() -> None:
    st.title("Ask a question")
    st.caption("Answers are grounded in our internal HR policies and SOPs — sources are cited below each response.")

    if "employee_chat_messages" not in st.session_state:
        st.session_state.employee_chat_messages = []

    for message in st.session_state.employee_chat_messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
            if message["role"] == "assistant":
                _render_citations(message.get("citations", []))

    question = st.chat_input("Ask about our HR policies, SOPs, or documents...")
    if not question:
        return

    st.session_state.employee_chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking..."):
            try:
                response = api_client.query(question)
            except Exception as exc:
                st.error(f"Couldn't reach the assistant: {exc}")
                return
        st.write(response.answer)
        _render_citations(response.citations)

    st.session_state.employee_chat_messages.append(
        {"role": "assistant", "content": response.answer, "citations": response.citations}
    )
