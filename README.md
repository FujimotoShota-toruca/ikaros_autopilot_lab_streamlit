# IKAROS：B-plane ダーツ v11

## v10からの修正
- `japanize_matplotlib` を削除（Python 3.13で distutils が無く落ちるため）
- 代わりに Matplotlib のシステムフォントから日本語対応フォント(Noto Sans CJK JP等)を自動選択
- v10で欠けていた `score_game()` を復活
- B-planeメイン、2D軌道図のノミナルは最終で金星位置に一致

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```
