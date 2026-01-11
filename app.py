"""
Streamlit エントリーポイント（UI層）

方針：
- 状態遷移は core/model.py
- 図は core/plots.py
- 設定は core/config.py
- フォントは core/fonts.py

という分割で “見通しの良さ” を優先しています。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import GameConfig
from core.fonts import setup_japanese_font, default_bundled_font_path
from core.model import build_sections, init_game, execute_section, score_game, comm_available, earth_angle_base_deg, predicted_earth_angle_deg, GameState
from core.plots import plot_bplane, plot_orbits_2d_nominal, plot_beta_maps, geometry_3d_figure


# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="IKAROS B-plane Darts v12", layout="wide")
st.title("🎯 IKAROS：B-plane ダーツ（適応誘導オペレーション）")
st.caption("v12：フォント同梱で日本語化／予測楕円の表示安定化／軌道図は曲線表示／コード分割＆日本語コメント多め。")


# -----------------------------
# フォント（同梱）
# -----------------------------
font_name, font_path = setup_japanese_font(default_bundled_font_path())


# -----------------------------
# 設定・初期化
# -----------------------------
cfg = GameConfig()
sections = build_sections()

with st.sidebar:
    st.header("設定")
    seed = st.number_input("シード（同じ問題を再現）", min_value=1, max_value=999999, value=42, step=1)
    show_truth = st.toggle("先生モード：真値を表示", value=False)

    st.divider()
    st.subheader("日本語フォント")
    if font_name:
        st.caption(f"同梱フォントを使用：{font_name}")
    else:
        st.warning("同梱フォントが読めず、日本語が□になる可能性があります。")

    st.divider()
    st.subheader("学習ポイント")
    st.markdown(
        """
- **SRPは弱い** → “調整” しかできない  
- **投入誤差**がある → 放置は負け筋  
- **通信ウィンドウ**は軌道幾何で決まる → βで“指向”を合わせる  
- でも βを増やすと **発電が落ちる**  
"""
    )


seed_int = int(seed)

# セッション状態（Streamlitの“擬似永続化”）
STATE_KEY = "bplane_state_v12"
SEED_KEY = "bplane_seed_v12"
PAGE_KEY = "page_v12"

if STATE_KEY not in st.session_state or st.session_state.get(SEED_KEY) != seed_int:
    st.session_state[STATE_KEY] = init_game(cfg, sections, seed=seed_int)
    st.session_state[SEED_KEY] = seed_int
    st.session_state[PAGE_KEY] = "Play"

state: GameState = st.session_state[STATE_KEY]


def rerun():
    # Streamlitのバージョン差分吸収
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())


def reset():
    st.session_state[STATE_KEY] = init_game(cfg, sections, seed=seed_int)
    st.session_state[PAGE_KEY] = "Play"
    rerun()


# 状態が result ならページも result に飛ばす
if state.phase == "result":
    st.session_state[PAGE_KEY] = "Result"

page = st.radio("ページ", ["Play", "Result"], horizontal=True, index=(0 if st.session_state[PAGE_KEY] == "Play" else 1))
st.session_state[PAGE_KEY] = page


# -----------------------------
# Play画面
# -----------------------------
def render_play():
    sec = sections[min(state.k, len(sections) - 1)]
    ea_base = earth_angle_base_deg(state, cfg, sections)
    ea = predicted_earth_angle_deg(state.beta_in, state.beta_out, state, cfg, sections)
    comm_ok = comm_available(state.beta_in, state.beta_out, state, cfg, sections)

    # 進捗
    st.progress(min(1.0, state.k / len(sections)))
    st.write(f"進捗：**{state.k}/{len(sections)}** セクション完了（全{len(sections)}）  |  現在：**{sec.name}**（t≈{sec.t_day:.0f}日）")

    # 上段メトリクス＋ボタン
    a1, a2, a3, a4, a5 = st.columns([1.0, 1.1, 1.1, 1.3, 1.5])
    with a1:
        st.metric("通信", "🟢OK" if comm_ok else "🔴NG")
    with a2:
        st.metric("バッテリ", f"{state.energy:.0f}/{cfg.energy_max:.0f}")
    with a3:
        st.metric("地球角(幾何)", f"{ea_base:+.1f}°")
    with a4:
        st.metric("地球角(指向後)", f"{ea:+.1f}°")
    with a5:
        btn_next = st.button("▶ このセクションを実行（進める）", use_container_width=True, disabled=(state.phase == "result"))
        btn_reset = st.button("🔁 リセット", use_container_width=True)

    if btn_reset:
        reset()
    if btn_next:
        execute_section(state, cfg, sections)
        rerun()

    # -------------------------
    # メイン：B-plane
    # -------------------------
    st.subheader("B-plane（メイン）")
    st.pyplot(plot_bplane(state, cfg, sections, show_truth=show_truth), use_container_width=True)

    if comm_ok:
        st.success("このβなら通信OK見込み（コマンド送信＆データ下ろし）。")
    else:
        st.warning("このβだと通信NG見込み → 実行するとΔβ=0固定＆DLできない。")

    # 左：軌道 右：マップ＋幾何＋コマンド
    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.subheader("位置関係（2D軌道図：ノミナル）")
        st.pyplot(plot_orbits_2d_nominal(state, cfg, sections), use_container_width=True)

        if state.log:
            df = pd.DataFrame(state.log)
            st.subheader("ライブ推移")
            st.line_chart(df.set_index("turn")[["dist_to_target_km"]], height=170)
            st.line_chart(df.set_index("turn")[["energy", "earth_angle_deg"]], height=200)

    with right:
        st.subheader("βin×βout マップ（幾何 + 指向 + 電力）")
        st.pyplot(plot_beta_maps(state, cfg, sections), use_container_width=True)

        st.subheader("幾何（3D表示）")
        st.caption("ドラッグで回転できます。")
        st.plotly_chart(geometry_3d_figure(state, cfg, sections), use_container_width=True)

        st.subheader("コマンド（βin / βout）")
        if not sec.uplink_possible:
            st.error("このセクションは NO-LINK：通信不可（コマンド固定）。")

        cA, cB = st.columns(2)
        with cA:
            bi = st.slider("βin [deg]", -35.0, 35.0, float(state.beta_in), 1.0)
        with cB:
            bo = st.slider("βout [deg]", -35.0, 35.0, float(state.beta_out), 1.0)

        state.beta_in = float(bi)
        state.beta_out = float(bo)

    # ログ表示（必要なら）
    if state.log:
        with st.expander("ログ（必要なら開く）", expanded=False):
            st.dataframe(pd.DataFrame(state.log), use_container_width=True, hide_index=True)


# -----------------------------
# Result画面
# -----------------------------
def render_result():
    st.header("📊 リザルト")
    score, bd = score_game(state, cfg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("スコア", f"{score:.0f}")
    c2.metric("最終距離（B-plane）", f"{bd['final_distance_km']:.0f} km")
    c3.metric("データ下ろし", f"{bd['science_downlinked']:.0f}")
    c4.metric("電力残", f"{bd['energy_left']:.0f}")

    st.subheader("B-plane（最終）")
    st.pyplot(plot_bplane(state, cfg, sections, show_truth=True), use_container_width=True)

    st.subheader("位置関係（2D軌道図：ノミナル）")
    st.pyplot(plot_orbits_2d_nominal(state, cfg, sections), use_container_width=True)

    if state.log:
        df = pd.DataFrame(state.log)
        st.subheader("推移まとめ")
        st.line_chart(df.set_index("turn")[["dist_to_target_km", "energy", "earth_angle_deg", "data_buffer", "data_lost_total"]], height=280)

    if st.button("🔁 もう一回（リセット）", use_container_width=True):
        reset()


if page == "Play":
    render_play()
else:
    render_result()
