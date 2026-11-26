"""Domain knowledge priors & RAG retrieval for printable characters and MakerWorld functional parts."""

from spoolgenie.knowledge.character_catalog import (
    CHARACTER_CATALOG,
    CharacterPreset,
    list_local_loras,
    match_character_catalog,
)
from spoolgenie.knowledge.kb import (
    CHARACTER_STYLE_PRESETS,
    KnowledgeEntry,
    PrintKnowledgeBase,
)

__all__ = [
    "CHARACTER_CATALOG",
    "CHARACTER_STYLE_PRESETS",
    "CharacterPreset",
    "KnowledgeEntry",
    "PrintKnowledgeBase",
    "list_local_loras",
    "match_character_catalog",
]
