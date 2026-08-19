import streamlit as st

from port6.schemas import DocumentMeta
from port6.ui import api_client


def _grouped_by_document(documents: list[DocumentMeta]) -> dict[str, list[DocumentMeta]]:
    grouped: dict[str, list[DocumentMeta]] = {}
    for doc in documents:
        grouped.setdefault(doc.document_id, []).append(doc)
    for versions in grouped.values():
        versions.sort(key=lambda d: d.version, reverse=True)
    return grouped


def _render_upload() -> None:
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload a policy document", type=["pdf"])
        if uploaded_file is None or not st.button("Ingest document", type="primary"):
            return

        with st.status(f"Ingesting {uploaded_file.name}...", expanded=True) as status:
            try:
                result = api_client.upload_document(uploaded_file.getvalue(), uploaded_file.name)
            except Exception as exc:
                status.update(label="Ingestion failed", state="error")
                st.error(str(exc))
                return

            if result["decision"] == "skip_duplicate":
                st.write("Already ingested — this exact file was uploaded before, nothing changed.")
                status.update(label="No changes (duplicate)", state="complete")
            else:
                kind = "New document" if result["decision"] == "new_document" else "New version"
                st.write(
                    f"{kind}: {result['num_parent_chunks']} parent chunk(s), "
                    f"{result['num_child_chunks']} child chunk(s) stored."
                )
                status.update(label="Ingestion complete", state="complete")


def _render_version_controls(document_id: str, versions: list[DocumentMeta]) -> None:
    with st.expander(f"{versions[0].source_filename}  ·  {len(versions)} version(s)"):
        for doc in versions:
            cols = st.columns([3, 2, 2, 2])
            cols[0].write(f"**v{doc.version}** — {doc.ingested_at.strftime('%Y-%m-%d')}")
            with cols[1]:
                if doc.active:
                    st.badge("Active", color="green")
                else:
                    st.badge("Inactive", color="gray")

            view_key = f"viewing_{document_id}_{doc.version}"
            if cols[2].button("View", key=f"view-{document_id}-{doc.version}"):
                st.session_state[view_key] = not st.session_state.get(view_key, False)
                st.rerun()

            if doc.active:
                if cols[3].button("Deactivate", key=f"deactivate-{document_id}-{doc.version}"):
                    api_client.deactivate_document(document_id)
                    st.rerun()
            elif cols[3].button("Reactivate", key=f"activate-{document_id}-{doc.version}"):
                api_client.activate_version(document_id, doc.version)
                st.rerun()

            if st.session_state.get(view_key):
                try:
                    file_bytes = api_client.get_document_file(document_id, doc.version)
                except Exception as exc:
                    st.error(f"Couldn't load the file: {exc}")
                else:
                    st.pdf(file_bytes, height=500)

        confirm_key = f"confirm_delete_{document_id}"
        if st.session_state.get(confirm_key):
            st.warning(f"Delete all {len(versions)} version(s) of this document permanently? This can't be undone.")
            confirm_cols = st.columns(2)
            if confirm_cols[0].button("Yes, delete permanently", key=f"confirm-yes-{document_id}", type="primary"):
                api_client.delete_document(document_id)
                st.session_state[confirm_key] = False
                st.rerun()
            if confirm_cols[1].button("Cancel", key=f"confirm-no-{document_id}"):
                st.session_state[confirm_key] = False
                st.rerun()
        elif st.button("Delete document", key=f"delete-{document_id}"):
            st.session_state[confirm_key] = True
            st.rerun()


def render() -> None:
    st.title("Manage documents")
    st.caption("Upload, version, and retire the policy documents the assistant answers from.")

    _render_upload()

    st.divider()
    st.subheader("Document library")

    try:
        documents = api_client.list_documents()
    except Exception as exc:
        st.error(f"Couldn't reach the assistant API: {exc}")
        return

    if not documents:
        st.info("No documents ingested yet.")
        return

    table_rows = [
        {
            "Document": doc.source_filename,
            "Version": doc.version,
            "Active": "Yes" if doc.active else "—",
            "Ingested": doc.ingested_at.strftime("%Y-%m-%d %H:%M"),
        }
        for doc in sorted(documents, key=lambda d: (d.document_id, -d.version))
    ]
    st.dataframe(table_rows, width="stretch", hide_index=True)

    st.subheader("Manage versions")
    for document_id, versions in _grouped_by_document(documents).items():
        _render_version_controls(document_id, versions)
