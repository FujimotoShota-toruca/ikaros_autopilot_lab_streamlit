# IKAROS：B-plane ダーツ（角度モデル版）

## 方針
- 通信/発電を **角度だけ**で定義
  - α = angle(n, 太陽方向 s) → 発電
  - γ = angle(n, 地球方向 e) → 通信（コーン内）
- βin/βout は「帆法線 n を作るつまみ」
- ノミナル軌道は u∈[0,1] の連続曲線で描画（直線結線をやめました）

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 日本語フォント
同梱フォント：NotoSansCJK-Regular.ttc

Streamlit Cloudで文字化けする場合は、assets/fonts に日本語フォント（ttf/otf/ttc）を追加してみてください。
