import json
import time
from dataclasses import dataclass
from pathlib import Path

from port6.eval.ir_metrics import citation_accuracy, hit_at_k, reciprocal_rank
from port6.eval.ragas_eval import evaluate_generation
from port6.rag.chain import answer_query
from port6.retrieval.reranker import get_reranking_retriever

DEFAULT_GOLDEN_SET_PATH = Path(__file__).resolve().parents[3] / "data" / "golden_eval_set.json"


@dataclass
class EvalResult:
    id: str
    question: str
    passed_refusal_check: bool
    answer: str
    latency_seconds: float
    hit: bool | None = None
    reciprocal_rank_score: float | None = None
    citation_correct: bool | None = None
    ragas_scores: dict[str, float] | None = None


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> list[dict]:
    return json.loads(path.read_text())


def run_eval(golden_set: list[dict] | None = None, run_ragas: bool = True) -> list[EvalResult]:
    """Runs every golden-set question through the live retrieval+chain and
    scores it. run_ragas=False skips the LLM-judged metrics (for cheap
    sweeps across many pipeline configs, e.g. in the Developer Playground)
    — IR metrics and citation-accuracy still run either way, since they're
    free."""
    golden_set = golden_set if golden_set is not None else load_golden_set()
    results: list[EvalResult] = []

    for item in golden_set:
        start = time.monotonic()
        response = answer_query(item["question"])
        latency = time.monotonic() - start

        result = EvalResult(
            id=item["id"],
            question=item["question"],
            passed_refusal_check=response.refused == item["expect_refusal"],
            answer=response.answer,
            latency_seconds=latency,
        )

        if not item["expect_refusal"]:
            retrieved = get_reranking_retriever().invoke(item["question"])
            result.hit = hit_at_k(retrieved, item["expected_document_id"])
            result.reciprocal_rank_score = reciprocal_rank(retrieved, item["expected_document_id"])
            result.citation_correct = citation_accuracy(
                response.citations, item["expected_document_id"], item["expected_page"]
            )

            if run_ragas and not response.refused:
                contexts = [c.snippet for c in response.citations] or [d.page_content for d in retrieved]
                try:
                    result.ragas_scores = evaluate_generation(
                        item["question"], response.answer, contexts, item["reference_answer"]
                    )
                except Exception as exc:
                    # A single question's LLM-judged scoring failing (e.g. a
                    # transient network error) shouldn't discard everything
                    # else in a multi-minute run — record it as missing and
                    # keep going; summarize() already treats None as "skip
                    # this one" when averaging.
                    print(f"RAGAS scoring failed for {item['id']!r}: {exc}")

        results.append(result)
        print(f"[{len(results)}/{len(golden_set)}] {item['id']} done ({latency:.1f}s)", flush=True)

    return results


def summarize(results: list[EvalResult]) -> dict:
    """Aggregate averages across a run — what the Developer Playground
    displays when comparing two pipeline configurations against each
    other."""
    n = len(results)
    refusal_accuracy = sum(r.passed_refusal_check for r in results) / n
    avg_latency = sum(r.latency_seconds for r in results) / n

    scored = [r for r in results if r.hit is not None]
    recall_at_k = sum(r.hit for r in scored) / len(scored) if scored else None
    mrr = sum(r.reciprocal_rank_score for r in scored) / len(scored) if scored else None
    citation_accuracy_rate = sum(r.citation_correct for r in scored) / len(scored) if scored else None

    ragas_results = [r.ragas_scores for r in results if r.ragas_scores]
    ragas_averages = None
    if ragas_results:
        ragas_averages = {key: sum(r[key] for r in ragas_results) / len(ragas_results) for key in ragas_results[0]}

    return {
        "n_questions": n,
        "refusal_accuracy": refusal_accuracy,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "citation_accuracy": citation_accuracy_rate,
        "avg_latency_seconds": avg_latency,
        "ragas": ragas_averages,
    }
