# IKAROS：B-plane ダーツ v12

## v12の変更点
- **日本語フォントを同梱**（assets/fonts/NotoSansCJK-Regular.ttc）→ Streamlit Cloudでも文字化けしにくい
- **予測楕円が時々出ない問題を緩和**：最小サイズを設定し、NaN/Infの共分散をサニタイズ
- **2D軌道図のノミナルを曲線表示**（uを細かくサンプリングして結線）
- **app.pyを分割**：core/config.py, core/model.py, core/plots.py, core/fonts.py
- **日本語コメント多め**（教材用に読みやすさ重視）

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 補足（ノミナルについて）
- 本教材のノミナル（計画）は「最終で金星位置に一致」するように定義した概念曲線です。
- “当たる/外れる”は B-plane側で表現し、軌道図は幾何（地球角・通信ウィンドウ）の直感理解に使います。
