# IKAROS Autopilot Lab (Streamlit)

宇宙少年団向けの、IKAROSを題材にした「太陽光帆＋姿勢制御」風のプログラミング教材アプリ（試作）です。  
低学年は手動で遊べて、高学年〜中学生はルール/PD/評価でやりごたえが出る構成にしています。

## ローカル実行
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud で公開（最短）
1. このフォルダを GitHub リポジトリに push
2. Streamlit Community Cloud で「New app」→ GitHub リポジトリを選択
3. Main file path に `app.py` を指定して Deploy

依存関係は `requirements.txt` で管理します。

## 安全性について
- 公開アプリで `exec/eval` による任意コード実行は危険なので、この試作では **MiniScript（簡易言語）** をパースして動かします。
- もし「参加者ごとにPythonを書かせたい」場合は、ブラウザ内実行（Pyodide など）や、隔離されたサンドボックスでの実行を検討してください。

## カスタマイズのポイント
- `missions()` のスコア係数・ノイズ・遅れ・外乱で難易度調整
- `step_sim()` の力/トルク式を変えると“機体っぽさ”が変わります
