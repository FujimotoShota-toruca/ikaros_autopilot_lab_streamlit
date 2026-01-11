# IKAROS：B-plane ダーツ v6

- βin×βout平面で、**電力収支（発電-消費）** と **ダウンリンク量（通信可否）** を可視化するマップを追加しました。
- β=0で勝ててしまう問題を解消するため、**投入誤差（初期B-planeズレ）** を導入し、
  さらに **推定が真値を勝手に動かしてしまう不具合** を修正しました。

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```
