# IKAROS：B-plane ダーツ v9

## 変更点
- B-planeはMatplotlib（文字・凡例が読みやすい）
- 位置関係を2D軌道図で表示（太陽・地球・金星・IKAROS）
- その幾何から地球角（ベース）を計算し、βの指向で通信可否が変わる
- βin/outマップはMatplotlibで可視化（通信OK領域は緑）
- 幾何はPlotly 3Dで表示（回転できる）

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```
