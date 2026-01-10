# IKAROS β-GO!（全部Vega v6）

v6の変更点：
- 軌道(2D)が“はっちゃける”問題を修正（Vega-Liteがxでソートして線を結んでしまうのが原因 → **order=day**を明示）
- β角は **スライダー＋直打ち（number_input）** を同期
- 予測を2本表示：
  1) 予測（β固定）
  2) 予測（ノミナルへ戻す）

実行:
```bash
pip install -r requirements.txt
streamlit run app.py
```
