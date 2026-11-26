"""MakerWorld search, reference image analysis, and local STL/3MF inspection."""

from spoolgenie.reference.makerworld import (
    analyze_reference_image,
    analyze_reference_mesh,
    inspect_makerworld_url,
    search_makerworld,
)

__all__ = [
    "search_makerworld",
    "inspect_makerworld_url",
    "analyze_reference_image",
    "analyze_reference_mesh",
]
