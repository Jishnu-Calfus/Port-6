from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from port6.config import settings
from port6.ingestion.loader import PageText

# Try paragraph breaks first, then lines, then sentences, then words — only
# fall back to a hard character cut if nothing else fits.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Joins pages before splitting. Deliberately NOT "\n\n" — that's the
# splitter's top-priority separator, so an artificial "\n\n" at every single
# page boundary would make it split there constantly regardless of whether
# the source content actually had a paragraph break at that page turn,
# quietly recreating the "overlap resets every page" problem this file
# exists to avoid. A plain space is far down the separator priority list, so
# it only gets used as a split point when nothing better is nearby.
PAGE_JOINER = " "


def get_parent_splitter() -> RecursiveCharacterTextSplitter:
    """Full-context chunks handed to the LLM at generation time. Sized to
    keep a whole policy clause (eligibility + exception + number) intact
    rather than split across chunk boundaries. add_start_index=True is what
    lets us recover which page a chunk actually came from after splitting."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        separators=SEPARATORS,
        add_start_index=True,
    )


def get_child_splitter() -> RecursiveCharacterTextSplitter:
    """Small chunks that actually get embedded — precise enough that a query
    only pulls in the specific sentence/clause it's about, not a whole
    section's worth of unrelated neighboring content. Splits within an
    already-small parent chunk, so it inherits that parent's page_number."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=SEPARATORS,
    )


def get_semantic_child_splitter(embeddings):
    """Playground-only alternative to get_child_splitter(): splits on
    embedding-similarity breakpoints instead of fixed size. Never used by the
    production ingestion pipeline (non-deterministic, slower) — exists so the
    Developer Playground can measure it against the recursive default."""
    from langchain_experimental.text_splitter import SemanticChunker

    return SemanticChunker(embeddings)


def _concat_pages(pages: list[PageText]) -> tuple[str, list[tuple[int, int]]]:
    """Join all pages into one string, recording the character offset each
    page starts at. Splitting the whole document at once — instead of one
    Document per page — lets chunk_overlap protect a paragraph that
    continues across a page break, rather than resetting at every boundary."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page in pages:
        if not page.text.strip():
            continue
        offsets.append((cursor, page.page_number))
        parts.append(page.text)
        cursor += len(page.text) + len(PAGE_JOINER)
    return PAGE_JOINER.join(parts), offsets


def _page_at_offset(offsets: list[tuple[int, int]], start_index: int) -> int:
    page_number = offsets[0][1]
    for offset, page in offsets:
        if offset > start_index:
            break
        page_number = page
    return page_number


def split_into_parents(pages: list[PageText], document_id: str) -> list[Document]:
    """The one function the ingestion pipeline calls: pages in, page-tagged
    parent chunks out, ready to hand to ParentDocumentRetriever (with
    parent_splitter=None, since splitting already happened here)."""
    full_text, offsets = _concat_pages(pages)
    if not full_text:
        return []

    splitter = get_parent_splitter()
    chunks = splitter.create_documents([full_text], metadatas=[{"document_id": document_id}])

    for chunk in chunks:
        start_index = chunk.metadata.get("start_index", 0)
        chunk.metadata["page_number"] = _page_at_offset(offsets, start_index)

    return chunks
