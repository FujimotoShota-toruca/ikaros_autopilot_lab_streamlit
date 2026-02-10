\
# IKAROS-GO（試作 v4）

Streamlitで動く、IKAROSっぽい「うごかして学ぶ」ゲームです。

## すぐ動かす（模型モード）
```bash
pip install -r requirements.txt
streamlit run app.py
```

## “IKAROSっぽい”データを自動生成（おすすめ）
```bash
python tools/generate_data.py --out data --profile ikaros2010 --step 2
streamlit run app.py
```

できるもの：
- `data/orbit_schedule.json`（太陽/地球/金星/IKAROS の 3D 位置）
- `data/sensitivity_schedule.json`（β→B-plane の効き 2×2）
- `data/mission_config.json`（ターン数・初期B-planeなど）

## タブ
- B-plane（ねらい）: 予測楕円つき
- 太陽系の図（2D）: いまどこ？
- βマップ: 通信できる場所 / でんき
- 3次元可視化: 太陽方向・地球方向・帆面法線（β）ベクトル表示


## v5 で追加
- B-planeに「金星（基準点）」「目標」「許容誤差（半径）」を表示
- 3Dタブを“近傍表示”に変更（軌道なし・通信可能コーン + 太陽/地球/帆ベクトル + 帆平面）


## v6 で追加（ゲーム性と見た目）
- 3Dタブ：通信コーンを帆面法線（アンテナ向き）に固定、地球/太陽ベクトルと一緒に表示
- βマップ：3Dベクトル（太陽方向・地球方向）を使って「でんき」「通信OK」を計算
- B-plane：ダーツ風の同心円とスコア表示（近いほど高得点）
- β=0でも少しずつズレる“ドリフト”を追加
- 初期の推定誤差を少し大きめに調整

- βマップに「通信の強さ(0-100)」の図も追加
