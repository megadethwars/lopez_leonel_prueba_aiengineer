"""Configuración por variables de entorno (ver .env.example).

Solo la lee `bootstrap.py` (composition root): los adapters reciben sus
parámetros explícitamente y el núcleo no conoce variables de entorno.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5-5"))
    anthropic_effort: str = field(default_factory=lambda: _env("ANTHROPIC_EFFORT", "low"))
    max_tokens: int = field(default_factory=lambda: int(_env("ANTHROPIC_MAX_TOKENS", "4096")))
    embedding_model: str = field(
        default_factory=lambda: _env(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    retrieval_top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "4")))
    retrieval_context_threshold: float = field(
        default_factory=lambda: float(_env("RETRIEVAL_CONTEXT_THRESHOLD", "0.12"))
    )
    retrieval_threshold: float = field(
        default_factory=lambda: float(_env("RETRIEVAL_THRESHOLD", "0.25"))
    )
    retrieval_alpha: float = field(default_factory=lambda: float(_env("RETRIEVAL_ALPHA", "0.7")))
    max_tool_iterations: int = field(default_factory=lambda: int(_env("MAX_TOOL_ITERATIONS", "4")))
    knowledge_base_dir: Path = field(
        default_factory=lambda: Path(_env("KNOWLEDGE_BASE_DIR", str(_REPO_ROOT / "data" / "knowledge_base")))
    )
    skills_dir: Path = field(default_factory=lambda: Path(_env("SKILLS_DIR", str(_REPO_ROOT / "skills"))))
    conversations_dir: Path = field(
        default_factory=lambda: Path(_env("CONVERSATIONS_DIR", str(_REPO_ROOT / "conversations")))
    )
    conversation_skill: str = field(
        default_factory=lambda: _env("CONVERSATION_SKILL", "atencion_al_cliente")
    )


settings = Settings()
