# IKAROS GO!（超シンプル版）

HTV GO! のように「説明が短くて、すぐ遊べる」路線に寄せた、IKAROS題材の教材ゲーム（試作）です。

## 特徴
- 3ステージ（短時間で終わる）
- 操作は **ハンドル1本**（-100〜100）
- 先生モードでだけ、ノイズ/外乱/制限時間/ログが出ます（通常は隠す）

## ローカル実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud で公開
GitHubに push → Streamlit Community Cloudで `app.py` を指定して Deploy。
