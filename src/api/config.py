from enum import Enum
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

# Retained as type definitions only. Neither is consulted at runtime: Qdrant is
# the sole backend (the FAISS embedder is archived under archive/legacy_2026_10/)
# and semantic chunking is the only ingestion entrypoint. Kept because they name
# the intended domain; they do not select behaviour.
class RetrievalBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"

class Settings(BaseSettings):
    # Only values that are actually read by the code belong here.
    #
    # `groq_api_key`, `qdrant_path`, `retrieval_backend`, `chunking_strategy`,
    # `data_path` and `default_top_k` were all removed. None was read anywhere
    # (0 references to settings.<name> across src/, scripts/ and tests/), because
    # their real consumers read the environment or use a different source:
    #
    #   GROQ_API_KEY      src/generation/generator.py    os.getenv("GROQ_API_KEY")
    #   QDRANT_PATH       src/retrieval/qdrant_store.py os.getenv("QDRANT_PATH", ...)
    #   top_k             PolicyEngine.retrieval_strategy() per query class
    #                      (5 factual / 8 general / 10 reasoning)
    #   chunking          src/ingestion/semantic_chunker.py is the entrypoint
    #   retrieval backend Qdrant is the only backend left; the FAISS embedder was
    #                      archived to archive/legacy_2026_10/
    #
    # Keeping them advertised a configuration surface that silently did nothing.
    # `default_top_k = 10` was actively misleading because the real default is
    # per query class, not 10.
    # model_config.env_file below means pydantic-settings already reads .env, so
    # the environment remains the single source for the values that moved.

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    bm25_chunks_path: str = "data/processed/chunks_semantic_v2.json"
    candidate_k: int = 50
    hybrid_rrf_k: int = 60
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices("RAG_RERANK_MODEL", "RERANKER_MODEL"),
    )
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("RAG_EMBED_MODEL", "EMBED_MODEL"),
    )
    score_gap_threshold: float = 15.0
    min_score_threshold: float = 0.0
    # Generation-time abstention gate. `generate()` defaulted to -3.0, which is
    # far more permissive than the 0.0 used by every other caller: with the
    # cross-encoder enabled a chunk scoring -0.195 is correctly rejected by the
    # CLI and Agent but was accepted by the API, which never passes the
    # argument. Measured: "How does hybrid search work?" returns top1 = -0.195
    # under rerank, so the API answered a question the other two paths abstained
    # on. 0.0 is the value check_retrieval_confidence() itself defaults to.
    retrieval_min_top1_score: float = 0.0
    # Confidence decision cuts (PROJECT_GOALS.md §5). These were literals in
    # ConfidenceEngine.__init__, so Phase 3 calibration required editing source
    # and Appendix A recorded them as not overridable. Same values, now
    # configurable. Adjusted per query type by TYPE_ADJUSTMENTS in
    # src/controller/confidence_engine.py.
    confidence_reject_threshold: float = 0.35
    confidence_partial_threshold: float = 0.65
    groq_model: str = "openai/gpt-oss-20b"
    groq_model_fast: str = "openai/gpt-oss-20b"
    generation_max_tokens: int = 768
    generation_temperature: float = 0.1
    generation_max_chars: int = 3000
    use_multi_query: bool = False
    use_hyde: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

settings = Settings()