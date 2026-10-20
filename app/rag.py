"""
Retrieval-augmented generation, used only on cache misses:

    query vector -> top-k knowledge-base chunks -> grounded prompt -> LLM

Deliberately separate from the cache. The cache is a shortcut that skips
generation entirely; RAG is what runs when there is no shortcut. Keeping
them as two stages makes each one easy to test and reason about.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.config import settings
from app.embeddings import Vector, get_embedder
from app.schemas import SourceChunk

KB_DIR = Path(__file__).parent / "knowledge_base"


@dataclass
class KBChunk:
    doc_id: str
    text: str
    vector: Vector


class KnowledgeBase:
    def __init__(self, directory: Path) -> None:
        self.chunks: list[KBChunk] = []
        self._matrix: npt.NDArray[np.float32] = np.zeros((0, 0), dtype=np.float32)
        self._load(directory)

    def _load(self, directory: Path) -> None:
        texts: list[str] = []
        doc_ids: list[str] = []
        for path in sorted(directory.glob("*.md")):
            content = path.read_text(encoding="utf-8").strip()
            # Paragraph-level chunks: simple, transparent, and each one is a
            # self-contained answer in this FAQ-style knowledge base.
            for i, para in enumerate(p.strip() for p in content.split("\n\n")):
                if para:
                    texts.append(para)
                    doc_ids.append(f"{path.name}#{i}")

        if not texts:
            return

        self._matrix = get_embedder().encode_batch(texts)
        self.chunks = [
            KBChunk(doc_id=d, text=t, vector=v)
            for d, t, v in zip(doc_ids, texts, self._matrix, strict=True)
        ]

    def retrieve(self, query_vector: Vector, top_k: int) -> list[SourceChunk]:
        if not self.chunks:
            return []
        scores = self._matrix @ query_vector
        top = np.argsort(-scores)[:top_k]
        return [
            SourceChunk(
                doc_id=self.chunks[i].doc_id,
                text=self.chunks[i].text,
                score=round(float(scores[i]), 4),
            )
            for i in top
        ]


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(KB_DIR)


def build_augmented_prompt(query: str, sources: list[SourceChunk]) -> str:
    context = "\n\n".join(f"[{s.doc_id}] {s.text}" for s in sources)
    return (
        "Answer the question using only the context below. If the context "
        "doesn't contain the answer, say so explicitly instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def retrieve_and_build_prompt(query: str, query_vector: Vector) -> tuple[str, list[SourceChunk]]:
    sources = get_knowledge_base().retrieve(query_vector, top_k=settings.rag_top_k)
    return build_augmented_prompt(query, sources), sources
