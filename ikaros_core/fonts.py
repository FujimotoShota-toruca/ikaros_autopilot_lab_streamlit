"""
フォント設定（Matplotlib）

assets/fonts にフォントを“同梱”し、それを優先的に使います。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
from matplotlib import font_manager as fm


def setup_japanese_font(font_path: str) -> Tuple[Optional[str], Optional[str]]:
    p = Path(font_path)
    if not p.exists():
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None, None

    try:
        fm.fontManager.addfont(str(p))
        name = fm.FontProperties(fname=str(p)).get_name()
        matplotlib.rcParams["font.family"] = name
        matplotlib.rcParams["axes.unicode_minus"] = False
        return name, str(p)
    except Exception:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None, None


def default_bundled_font_path() -> str:
    base = Path("assets") / "fonts"
    if not base.exists():
        return str(base / "NotoSansCJK-Regular.ttc")

    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in [".ttf", ".otf", ".ttc"]:
            return str(p)
    return str(base / "NotoSansCJK-Regular.ttc")
