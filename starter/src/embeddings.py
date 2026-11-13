"""Real local embedding provider via sentence-transformers (model chosen in config/strategies.json)."""
from typing import Any, Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class SentenceTransformerEmbedder:
    """Local model; raises on failure and never falls back to mock vectors.

    query_prompt / document_prompt are prepended exactly as given (model-card instructions),
    e.g. EmbeddingGemma 2: "task: search result | query: " and "title: none | text: ".
    """

    def __init__(self, model_name: str, query_prompt: str = "", document_prompt: str = "",
                 config_kwargs: dict[str, Any] | None = None, batch_size: int = 16,
                 truncate_dim: int | None = None, max_tokens: int | None = None,
                 revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        import torch

        kwargs: dict[str, Any] = {"model_kwargs": {"dtype": torch.float32}}  # bf16 on CPU is slow/lossy
        if config_kwargs:
            kwargs["config_kwargs"] = config_kwargs
        if revision:
            kwargs["revision"] = revision  # pin the exact model snapshot for reproducibility
        self.model = SentenceTransformer(model_name, device="cpu", **kwargs)
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dimensions = int(truncate_dim or get_dim())
        self.max_seq_length = int(min(self.model.max_seq_length or 0, max_tokens or 10**9))

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(self.document_prompt + text)["input_ids"])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False,
                                 truncate_dim=self.truncate_dim)
        out = vecs.tolist()
        if any(len(v) != self.dimensions for v in out):
            raise RuntimeError("embedding dimension mismatch from model")
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.document_prompt + t for t in texts])

    def embed_query(self, question: str) -> list[float]:
        return self._encode([self.query_prompt + question])[0]


def from_config(cfg: dict[str, Any]) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        cfg["model"], cfg.get("query_prompt", ""), cfg.get("document_prompt", ""),
        cfg.get("config_kwargs"), cfg.get("batch_size", 16), cfg.get("truncate_dim"),
        cfg.get("max_tokens"), cfg.get("model_revision"))
