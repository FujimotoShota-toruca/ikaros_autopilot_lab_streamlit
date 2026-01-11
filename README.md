# IKAROS：B-plane ダーツ（適応誘導オペレーション）

ISSFD 2011の論文にあるIKAROS誘導の本質（制約・不確かさ・OD更新・後半勝負）を、
B-plane上の“的当て”としてゲーム化したプロトタイプです。

## 実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 操作
- βin/βout を入力（スライダー＋直打ち）
- 「このセクションを実行」で2週間ぶん進む（7セクションで終了）
- NO-LINKセクションはコマンド不可（Δβ=0固定）
