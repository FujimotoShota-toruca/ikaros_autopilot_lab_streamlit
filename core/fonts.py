"""
フォント設定（Matplotlib）

Streamlit Cloud / Python 3.13 環境だと、
- OSに日本語フォントが入っていない
- あるいは入っていても名前が変わる
ことがあり、文字化け（□）が起きがちです。

そこで、このプロジェクトでは
  assets/fonts/NotoSansCJK-Regular.ttc
を“同梱”して、必ずそれを使うようにします。

※ このフォントは Noto Sans CJK（Google）で、一般に再配布可能なライセンスです。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
from matplotlib import font_manager as fm


def setup_japanese_font(font_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    font_path のフォントを Matplotlib に登録し、デフォルトフォントに設定する。
    """
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
    # app.py からの相対で呼ばれる想定（カレント = プロジェクトルート）
    return os.path.join("assets", "fonts", "NotoSansCJK-Regular.ttc")
