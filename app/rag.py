"""
Genuine RAG path, used only on cache misses:

  query -> embed -> retrieve top-k relevant chunks from the knowledge base
        -> build an augmented prompt (query + retrieved context)
        -> send to the LLM provider
        -> cache the (query, answer) pair for next time

This is deliberately separate from the cache layer. The cache is a
shortcut that bypasses generation entirely on a hit; RAG is what happens
when there's no shortcut and a real answer has to be generated. Keeping
them as two distinct stages is what makes each one easy to reason about
and to explain.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import settings
from app.embeddings import cosine_similarity, get_embedder
from app.schemas import SourceChunk

KB_DIR = Path(__file__).parent / "knowledge_base"


@dataclass
class KBChunk:
    doc_id: str
    text: str
    vector: np.ndarray


class KnowledgeBase:
    def __init__(self, directory: Path):
        self.chunks: list[KBChunk] = []
        self._load(directory)

    def _load(self, directory: Path) -> None:
        embedder = get_embedder()
        texts, doc_ids = [], []

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".md"):
                continue
            path = directory / filename
            content = path.read_text(encoding="utf-8").strip()
            # Split each file into paragraph-level chunks — simple and
            # transparent, appropriate for a knowledge base this small.
            for i, para in enumerate(p.strip() for p in content.split("\n\n")):
                if para:
                    texts.append(para)
                    doc_ids.append(f"{filename}#{i}")

        if not texts:
            return

        vectors = embedder.encode_batch(texts)
        self.chunks = [
            KBChunk(doc_id=doc_ids[i], text=texts[i], vector=vectors[i])
            for i in range(len(texts))
        ]

    def retrieve(self, query_vector: np.ndarray, top_k: int) -> list[SourceChunk]:
        scored = [
            (chunk, cosine_similarity(query_vector, chunk.vector))
            for chunk in self.chunks
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            SourceChunk(doc_id=c.doc_id, text=c.text, score=round(s, 4))
            for c, s in scored[:top_k]
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


def answer_with_rag(query: str, query_vector: np.ndarray):
    kb = get_knowledge_base()
    sources = kb.retrieve(query_vector, top_k=settings.rag_top_k)
    prompt = build_augmented_prompt(query, sources)
    return prompt, sources
