# IKAROS：B-plane ダーツ v8

## 追加/改善
- B-planeの文字を見やすく（白字 + 黒縁）
- 囲い（ターゲット円・制御範囲・予測範囲）に凡例（線レジェンド）
- 予測を点→範囲（楕円：1σ）で表示
- βマップの通信領域を“緑の面＋境界線”で表示
- 幾何（概念図）：太陽光・地球方向・帆法線（βeff）の関係

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```
