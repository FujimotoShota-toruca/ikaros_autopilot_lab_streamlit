\
# IKAROS-GO（試作）

Streamlitで動く、IKAROSっぽい「うごかして学ぶ」ゲームの試作品です。

- **β_in / β_out**（ベータ）を動かして、IKAROSの向きを変えます
- すると、金星での **B-plane（ねらいの平面）** の点が少しずつ動きます
- でも、いつも自由にできるわけではなくて…
  - **でんき（太陽からのかたむき）**
  - **通信（地球に向けられる角度）**
  がじゃまをします（本物の運用っぽさ）

> これは “教育用のカンタン模型” です。あとからあなたの計算したデータに差し替えて、本物っぽくできます。

---

## すぐ動かす（ローカル）

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Streamlit Community Cloudでデプロイ（はやい）

1. このフォルダを GitHub にアップロード
2. Streamlit Community Cloud で New app → Repo を選ぶ
3. **Main file path** を `app.py` にする
4. Deploy

---

## 画面の見かた（小学生むけ）

- **B-plane（ねらい）**  
  点が “いまの場所（よそう）”。  
  楕円が “つぎに行きそうな ばらつき（よそう）”。

- **太陽系の図（いまどこ？）**  
  太陽・地球・金星・IKAROS が 2D 図で見えます。  
  ※これは “それっぽい模型” の図です。

- **βマップ（通信とでんき）**  
  βの場所ごとに、
  - でんき（背景）
  - 通信できる場所（太い線）
  が見えます。

---

## 参考（読み物）

- ISSFD 2011: IKAROS の金星B-planeターゲティング運用など  
  https://issfd.org/ISSFD_2011/S3-Interplanetary.Mission.Design.1-IMD1/S3_P6_ISSFD22_PF_075.pdf

- J-STAGE: IKAROS の通信・アンテナ運用など  
  https://www.jstage.jst.go.jp/article/kjsass/61/4/61_KJ00008636303/_pdf/-char/ja

---

## ライセンス

MIT（`LICENSE` を見てください）
