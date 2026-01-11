# IKAROS：B-plane ダーツ（適応誘導オペレーション） v4

v3で発生した Altair の SchemaValidationError を回避するため、**Altairを廃止**し、
`st.vega_lite_chart()` に **Vega-Lite の dict spec を直接渡す**構成に変更しました。

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 変更点
- ✅ すべてのグラフを Vega-Lite（dict spec）で描画
- ✅ Python 3.13 dataclass の mutable default 問題を修正済み
