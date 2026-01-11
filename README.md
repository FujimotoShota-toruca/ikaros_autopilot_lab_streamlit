# IKAROS：B-plane ダーツ（適応誘導オペレーション） v3

v2で発生したエラー：

- `ValueError: mutable default ... use default_factory`

Python 3.13 の dataclass では `np.array(...)` のような **mutable default** が禁止なので、
`GameConfig.target` を `default_factory` に変更して修正しています。

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```
