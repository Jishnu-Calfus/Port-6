import streamlit as st

from port6.ui import api_client
from port6.ui.views import developer_view, employee_view, hr_view

st.set_page_config(page_title="Port6 Document Assistant", page_icon="📄", layout="wide")

with st.sidebar:
    st.title("Port6")
    st.caption("Internal Document Assistant")
    try:
        active_docs = {doc.document_id for doc in api_client.list_documents() if doc.active}
        st.metric("Active documents", len(active_docs))
    except Exception:
        st.warning("API unreachable — is the backend running?")
    st.divider()

employee_page = st.Page(employee_view.render, title="Employee", icon="💬", url_path="employee", default=True)
hr_page = st.Page(hr_view.render, title="HR", icon="🗂️", url_path="hr")
developer_page = st.Page(developer_view.render, title="Developer", icon="🧪", url_path="developer")

navigation = st.navigation([employee_page, hr_page, developer_page])
navigation.run()
