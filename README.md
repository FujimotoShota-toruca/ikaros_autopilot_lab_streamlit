# IKAROS β-GO!（2D・角度1本）

HTV GO!! っぽく「軌道は自動で進む」「操作は最小」を狙った、IKAROS題材の教材アプリ（試作）です。

## 概要
- 太陽中心の2D
- 地球・金星は円軌道として自動で動く（解析的に計算）
- プレイヤーは **β角（1本）**だけを操作して、金星に近づく

## モード
- **リアルタイム制御**：βをその場で変更して誘導
- **事前角度指定**：βを3区間だけ決めて一気に実行
- **エクストラ：時間無制限**：練習用（失敗なし）

## ローカル実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud で公開
このフォルダを GitHub に push → Streamlit Community Cloud で `app.py` を指定して Deploy。

## 注意
- 教材用の“おもちゃモデル”です（実際の航法設計に使うものではありません）。
- `streamlit_autorefresh` を入れるとリアルタイムの自動進行が滑らかになりますが、なくても「ちょっと進める」ボタンで遊べます。
