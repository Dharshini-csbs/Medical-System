from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union


@dataclass
class RagDocument:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaissRAG:
    """FAISS-backed retriever for medical knowledge snippets."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model = embedding_model
        self._encoder = None
        self._index = None
        self._documents: List[RagDocument] = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    def _embed(self, texts: Sequence[str]):
        import numpy as np

        embeddings = self._get_encoder().encode(list(texts))
        return np.asarray(embeddings, dtype="float32")

    def build_index(self, documents: Sequence[Union[RagDocument, str]]) -> None:
        import faiss

        normalized_docs: List[RagDocument] = []
        for item in documents:
            if isinstance(item, RagDocument):
                normalized_docs.append(item)
            else:
                normalized_docs.append(RagDocument(text=item))

        if not normalized_docs:
            raise ValueError("At least one document is required to build the index.")

        vectors = self._embed([doc.text for doc in normalized_docs])
        self._index = faiss.IndexFlatL2(vectors.shape[1])
        self._index.add(vectors)
        self._documents = normalized_docs

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self._index is None:
            raise ValueError("Index is not built yet. Call build_index() first.")

        query_vector = self._embed([query])
        distances, indices = self._index.search(query_vector, top_k)

        results: List[Dict[str, Any]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self._documents[idx]
            results.append({"text": doc.text, "metadata": doc.metadata, "distance": float(distance)})
        return results

