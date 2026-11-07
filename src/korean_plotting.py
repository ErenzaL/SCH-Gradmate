"""Matplotlib Korean font discovery and configuration helpers."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from matplotlib import font_manager


def _font_preferences(system: str | None = None) -> list[str]:
    """Return Korean font preferences for an OS, followed by portable fallbacks."""
    os_name = (system or platform.system()).lower()
    if os_name == "darwin":
        preferred = ["AppleGothic", "Apple SD Gothic Neo"]
    elif os_name == "windows":
        preferred = ["Malgun Gothic"]
    elif os_name == "linux":
        preferred = ["NanumGothic", "Nanum Gothic", "Noto Sans CJK KR", "Noto Sans CJK"]
    else:
        preferred = []
    portable = [
        "NanumGothic", "Nanum Gothic", "Noto Sans CJK KR", "Noto Sans CJK",
        "AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic",
    ]
    return list(dict.fromkeys(preferred + portable))


def _usable_font_path(font_name: str) -> str | None:
    """Find a real font file that contains common Hangul glyphs."""
    family_names = {entry.name.casefold() for entry in font_manager.fontManager.ttflist}
    if font_name.casefold() not in family_names:
        return None
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=[font_name]), fallback_to_default=False
        )
        if not Path(path).is_file():
            return None
        charmap = font_manager.get_font(path).get_charmap()
        if not all(ord(char) in charmap for char in "한글졸업요건"):
            return None
        return path
    except (ValueError, OSError, RuntimeError):
        return None


def configure_korean_font(plt: Any = None, system: str | None = None) -> str:
    """Configure Matplotlib with an installed Korean font and return its family.

    ``system`` is an optional OS override useful for testing platform preference
    order. A clear error is raised if no installed candidate can render Hangul.
    """
    if plt is None:
        import matplotlib.pyplot as plt

    for family in _font_preferences(system):
        path = _usable_font_path(family)
        if path:
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            return family
    raise RuntimeError(
        "한글을 지원하는 Matplotlib 글꼴을 찾지 못했습니다. "
        "macOS는 AppleGothic, Windows는 맑은 고딕, Linux는 나눔고딕 또는 Noto Sans CJK를 설치해 주세요."
    )
