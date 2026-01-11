# IKAROS：B-plane ダーツ v10

## 変更点
- 2D軌道図：ノミナル(計画)が最終で金星位置に一致するように補間を修正
- Matplotlib日本語化：japanize_matplotlib を導入（文字化け対策）
- コマンド：直打ちを削除し、スライダーのみ
- 画面構成：B-planeをメイン表示（最上段・大きく）
- 幾何：Plotly 3Dで回転表示

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```
