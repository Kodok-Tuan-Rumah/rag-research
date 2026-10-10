from enum import Enum
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

class RetrievalBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"

class Settings(BaseSettings): 
    api_host: str = "0.0.0.0"
    api_port: int = 8003
    retrieval_backend: RetrievalBackend = RetrievalBackend.QDRANT
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    data_path: Path = Path("data/processed")
    qdrant_path: Path = Path("data/qdrant_storage")
    bm25_chunks_path: str = "data/processed/chunks_semantic_v2.json"
    default_top_k: int = 10
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
    groq_api_key: str = ""
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