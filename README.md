# IKAROS-GO! v12（本格・決定論）

この版は「本格っぽさ」を優先した模型です。

- **状態遷移行列（感度行列）**っぽいもの `C(k)` を使って、B-plane上の位置を動かします
- **発電（太陽）**と **通信（地球）** を、姿勢（β_in/out）から判定します
- **乱数なし**：同じ操作なら同じ結果になります（再現性100%）

Build: v12.1-full-deterministic-2026-02-14

## 起動方法
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ざっくり数式（模型）
```
x_(k+1) = x_k + drift(k) + k · C(k) · [β_in, β_out]^T
```
- drift(k)：β=0でもズレる（決定論的）
- k：効きの誤差（真値k_trueと想定k_hat）
- 通信できたら推定が一気に良くなる（測位アップデートの模型）
