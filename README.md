# IKAROS-GO（試作 v3）

この版は **あなたの計算したデータをJSONで差し替え** できるようにしてあります。

## すぐ動かす

```bash
pip install -r requirements.txt
streamlit run app.py
```

## データを入れる（ここが本題）

### 1) 2D軌道：`data/orbit_schedule.json`
このファイルがあると、2D図と幾何（太陽方向/地球方向）がそのデータになります。

例：

```json
[
  {"day":0,  "sun":[0,0], "earth":[1,0], "venus":[0.723,0], "ikaros":[0.95,0.10]},
  {"day":14, "sun":[0,0], "earth":[0.97,0.24], "venus":[0.68,0.25], "ikaros":[0.90,0.20]}
]
```

### 2) 感度行列：`data/sensitivity_schedule.json`
このファイルがあると、β→B-plane の効きがそのデータになります。

例：

```json
[
  {"turn":0, "C":[[1.0,0.3],[-0.2,0.9]]},
  {"turn":1, "C":[[1.1,0.2],[-0.1,1.0]]}
]
```

まずは `data/*.example.json` をコピーして作るのが早いです。
