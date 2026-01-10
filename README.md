# IKAROS β-GO!（次の一手：全部Vega v4）

- Matplotlibなし（日本語文字化けしにくい）
- メイン図は **軌道誤差**（実・予測・ノミナル(0)・許容帯・通信区間を重ねる）
- 常時表示：発電量／地球角／軌道誤差／SRPの小ささ（μm/s² + ΔV）／燃料
- 予測が制限時間以降で変になる問題：**ノミナルを制限時間+余白まで事前計算**して修正

実行:
```bash
pip install -r requirements.txt
streamlit run app.py
```
