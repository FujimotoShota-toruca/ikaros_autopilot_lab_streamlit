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
