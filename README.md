# IKAROS β-GO!（全部Vega v5）

- **numpyのtrapz依存を排除**（Streamlit Cloudのnumpyビルド差でエラーになる対策）
- **軌道（2D）** をAltairで復活（実・ノミナル・予測＋地球/金星の円軌道）
- メイン図は **軌道誤差**（実・予測・ノミナル(0)・許容帯・通信区間）
- SRPの小ささ：**SRP加速度（μm/s²）** と **ΔV（積分）**
- 初期誤差（位置/速度/β）を調整できる（ゲームバランス用）

実行:
```bash
pip install -r requirements.txt
streamlit run app.py
```
