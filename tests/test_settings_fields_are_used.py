"""Every Settings field must be read somewhere.

`Settings` had two fields that nothing read:

    groq_api_key     consumers read os.getenv("GROQ_API_KEY") instead
    qdrant_path      consumers read os.getenv("QDRANT_PATH", ...) instead

Both were removed. They advertised a configuration surface that silently did
nothing — setting GROQ_API_KEY on Settings had no effect, so a reader could
reasonably believe the key was configurable there when it was not.

This module checks the general rule rather than those two names, so the next dead
field is caught when it is added instead of during a later audit.

Two fields are documented exceptions because their value is consumed
positionally or through the env-file mechanism rather than as
`settings.<name>`.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
CONFIG = SRC / "api" / "config.py"

# Fields whose presence is deliberate even though no `settings.<name>` read exists.
ALLOWED_UNREAD = {
    # Present so Settings() fails loudly on a malformed .env, and so the value is
    # documented; the actual key is read from the environment by the generator.
    "model_config",
}


def _searchable_files():
    files = [p for p in SRC.rglob("*.py") if p.name != "__init__.py"]
    for extra in ("scripts", "tests"):
        d = REPO_ROOT / extra
        if d.exists():
            files += [p for p in d.rglob("*.py") if p.name != "__init__.py"]
    return files


def _settings_fields() -> list[str]:
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Settings")
    return [
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    ]


def test_settings_has_fields():
    assert len(_settings_fields()) > 10, "Settings field discovery failed"


@pytest.mark.parametrize("field", _settings_fields())
def test_settings_field_is_read(field):
    """A field nobody reads is a configuration surface that does nothing."""
    if field in ALLOWED_UNREAD:
        pytest.skip(f"{field} is a documented exception")
    pattern = rf"\bsettings\.{re.escape(field)}\b"
    reads = 0
    for path in _searchable_files():
        if path == CONFIG:
            continue
        reads += len(re.findall(pattern, path.read_text(encoding="utf-8", errors="ignore")))
    assert reads > 0, (
        f"Settings.{field} is never read. Either wire it up or remove it — an "
        f"unread field advertises configuration that has no effect."
    )


def test_removed_dead_fields_are_absent():
    """Guard the specific regression: these two were dead config."""
    fields = set(_settings_fields())
    assert "groq_api_key" not in fields, (
        "groq_api_key must not reappear — the generator reads GROQ_API_KEY from "
        "the environment and a Settings field would shadow it with no effect"
    )
    assert "qdrant_path" not in fields, (
        "qdrant_path must not reappear — qdrant_store reads QDRANT_PATH from the "
        "environment and a Settings field would advertise a path that is ignored"
    )


def test_environment_remains_the_source_for_both():
    """If Settings drops them, the environment path must be the live one."""
    gen = (SRC / "generation" / "generator.py").read_text(encoding="utf-8")
    store = (SRC / "retrieval" / "qdrant_store.py").read_text(encoding="utf-8")
    assert 'os.getenv("GROQ_API_KEY")' in gen, "generator must read GROQ_API_KEY from env"
    assert 'os.getenv("QDRANT_PATH"' in store, "qdrant_store must read QDRANT_PATH from env"


def test_settings_still_loads_env_file():
    """Removing the fields must not break the env-file mechanism they relied on."""
    import src._compat  # noqa: F401
    from src.api.config import settings

    assert settings.groq_model
    assert settings.model_config.get("env_file") == ".env"