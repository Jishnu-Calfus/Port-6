import pandas as pd
import streamlit as st

from port6.eval.runner import load_golden_set
from port6.eval.strategy_comparison import (
    StrategyResult,
    compare_hnsw_configs,
    compare_retrieval_strategies,
    compare_vector_backends,
)
from port6.retrieval.reranker import get_reranking_retriever

_HNSW_PARAM_INFO = [
    ("M (max_neighbors)", "How many graph connections each vector keeps. Higher = more accurate search, but more memory and a slower one-time build."),
    ("ef_construction", "How thoroughly the graph is built when chunks are first indexed. Higher = better graph quality, slower ingestion (paid once, at upload time)."),
    ("ef_search", "How many candidates are checked per query at search time. Higher = better recall, slower per-query latency (paid on every question asked)."),
]

_RAGAS_METRIC_INFO = [
    ("Faithfulness", "Does the answer only state things actually supported by the retrieved text? Higher = less hallucination."),
    (
        "Answer Relevancy",
        "Does the answer actually address what was asked, without padding or drifting off-topic? Higher = more focused.",
    ),
    (
        "Context Precision",
        "Of the chunks retrieval pulled in, how much was actually useful for answering? Higher = less irrelevant noise.",
    ),
    (
        "Context Recall",
        "Did retrieval bring back everything needed to fully answer, or was something missing? Higher = more complete retrieval.",
    ),
]

# Computed once against the current 5-document corpus — one representative
# question per document, plus one extra from leave-policy. RAGAS's
# LLM-judge calls are too slow/rate-limited right now to run live from a
# button click the way the other tabs do (observed anywhere from ~5s to
# 300s+ per question depending on API load at the time) — this is a real,
# computed-once snapshot, not fabricated data, just not re-run on every
# page load. Covers all 5 documents.
_RAGAS_SAMPLE_SCORES: dict[str, dict[str, float]] = {
    "q01": {"faithfulness": 1.0, "answer_relevancy": 0.8113353805271193, "context_precision": 0.9999999999, "context_recall": 1.0},
    "q04": {"faithfulness": 0.36363636363636365, "answer_relevancy": 0.7283738313857357, "context_precision": 0.9999999999, "context_recall": 0.5},
    "q07": {"faithfulness": 0.5, "answer_relevancy": 0.9747280699411629, "context_precision": 0.99999999995, "context_recall": 1.0},
    "q11": {"faithfulness": 0.75, "answer_relevancy": 0.8910406116863253, "context_precision": 0.9999999999, "context_recall": 1.0},
    "q21": {"faithfulness": 1.0, "answer_relevancy": 0.80985315931438, "context_precision": 0.9999999999, "context_recall": 0.0},
    "q24": {"faithfulness": 1.0, "answer_relevancy": 0.7385114807269367, "context_precision": 0.9999999999, "context_recall": 1.0},
}


def _results_to_df(results: list[StrategyResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Strategy": r.name,
                "Recall@k": r.recall_at_k,
                "MRR": r.mrr,
                "Latency (ms)": r.avg_latency_seconds * 1000,
            }
            for r in results
        ]
    ).set_index("Strategy")


def _render_comparison(title: str, description: str, run_fn, state_key: str) -> None:
    st.subheader(title)
    st.caption(description)

    if st.button("Run comparison", key=f"run-{state_key}", type="primary"):
        with st.spinner("Running against the live golden eval set..."):
            st.session_state[state_key] = run_fn()

    results = st.session_state.get(state_key)
    if not results:
        st.info("Click the button to run this comparison against the real corpus and golden eval set — no canned numbers.")
        return

    df = _results_to_df(results)
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Accuracy (each metric is 0-1, shown grouped — not stacked)")
        st.bar_chart(df[["Recall@k", "MRR"]], stack=False)
    with col2:
        st.caption("Latency (ms)")
        st.bar_chart(df[["Latency (ms)"]])
    st.dataframe(df.reset_index(), width="stretch", hide_index=True)


def _render_retrieval_debug() -> None:
    st.subheader("Retrieval debug")
    st.caption("See exactly what the production pipeline (hybrid search + rerank) retrieves for any query.")

    query = st.text_input("Try a query", placeholder="e.g. What does SOP-114 require?")
    if not query:
        return

    with st.spinner("Retrieving..."):
        docs = get_reranking_retriever().invoke(query)

    if not docs:
        st.warning("No chunks retrieved for this query.")
        return

    for i, doc in enumerate(docs, start=1):
        with st.container(border=True):
            st.markdown(f"**#{i} — {doc.metadata.get('document_id')}** · page {doc.metadata.get('page_number')}")
            st.caption(doc.page_content)


def _render_hnsw_comparison() -> None:
    st.subheader("HNSW parameter sweep (M / ef_construction / ef_search)")
    st.caption("How much do the HNSW graph parameters actually move recall and latency at this scale?")

    for name, description in _HNSW_PARAM_INFO:
        st.markdown(f"**{name}** — {description}")

    if st.button("Run comparison", key="run-hnsw", type="primary"):
        with st.spinner("Building throwaway HNSW indexes and running against the live golden eval set..."):
            st.session_state["playground_hnsw_results"] = compare_hnsw_configs()

    results = st.session_state.get("playground_hnsw_results")
    if not results:
        st.info("Click the button to run this comparison against the real corpus and golden eval set — no canned numbers.")
        return

    df = pd.DataFrame(
        [
            {
                "Config": r.name,
                "M": r.m,
                "ef_construction": r.ef_construction,
                "ef_search": r.ef_search,
                "Recall@k": r.recall_at_k,
                "MRR": r.mrr,
                "Latency (ms)": r.avg_latency_seconds * 1000,
                "Trade-off": r.tradeoff,
            }
            for r in results
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Accuracy (each metric is 0-1, shown grouped — not stacked)")
        st.bar_chart(df.set_index("Config")[["Recall@k", "MRR"]], stack=False)
    with col2:
        st.caption("Latency (ms)")
        st.bar_chart(df.set_index("Config")[["Latency (ms)"]])

    st.dataframe(df, width="stretch", hide_index=True)


def _render_ragas_evals() -> None:
    st.subheader("Answer quality (RAGAS)")
    st.caption(
        "The other tabs measure whether retrieval found the right chunk. These metrics judge the final written "
        "answer itself, using an LLM as judge. This is a computed-once snapshot, not a live button — RAGAS's "
        "judge calls are currently too slow/rate-limited to re-run interactively (a single question has ranged "
        "from ~5s to 300s+ depending on API load)."
    )

    for name, description in _RAGAS_METRIC_INFO:
        st.markdown(f"**{name}** — {description}")

    if not _RAGAS_SAMPLE_SCORES:
        st.info("No snapshot recorded yet.")
        return

    golden_by_id = {g["id"]: g for g in load_golden_set()}
    rows = [
        {
            "Question": golden_by_id.get(qid, {}).get("question", qid),
            "Faithfulness": scores["faithfulness"],
            "Answer Relevancy": scores["answer_relevancy"],
            "Context Precision": scores["context_precision"],
            "Context Recall": scores["context_recall"],
        }
        for qid, scores in _RAGAS_SAMPLE_SCORES.items()
    ]
    df = pd.DataFrame(rows)
    averages = df.drop(columns="Question").mean().to_frame(name="Average score")

    st.bar_chart(averages)
    st.dataframe(df, width="stretch", hide_index=True)


def render() -> None:
    st.title("Developer playground")
    st.caption(
        "Prove what each advanced technique actually buys you — every comparison below runs live against the "
        "real corpus and the golden eval set, not canned numbers."
    )

    debug_tab, strategy_tab, backend_tab, hnsw_tab, evals_tab = st.tabs(
        ["Retrieval debug", "Retrieval strategy", "Vector backend", "HNSW tuning", "Evals"]
    )

    with debug_tab:
        _render_retrieval_debug()

    with strategy_tab:
        _render_comparison(
            "Naive vs. hybrid vs. hybrid + rerank",
            "Does hybrid search and reranking actually beat naive dense-only retrieval on this corpus?",
            compare_retrieval_strategies,
            "playground_strategy_results",
        )

    with backend_tab:
        _render_comparison(
            "Chroma vs. FAISS",
            "At this corpus's scale, does swapping the vector store backend change anything?",
            compare_vector_backends,
            "playground_backend_results",
        )

    with hnsw_tab:
        _render_hnsw_comparison()

    with evals_tab:
        _render_ragas_evals()
