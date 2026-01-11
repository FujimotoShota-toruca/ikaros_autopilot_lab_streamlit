"""
Streamlit エントリーポイント（UI層）

構成（可読性重視）
- core/config.py    : パラメータ
- core/attitude.py  : 角度モデル（n,s,eとα,γ）
- core/model.py     : 状態遷移（運用・OD・リソース）
- core/plots.py     : 図（B-plane / βマップ / 軌道 / 3D）
- core/fonts.py     : 日本語フォント

“角度だけ”で通信・発電を定義したバージョンです。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # Streamlit上での描画安定化

from ikaros_core.config import GameConfig
from ikaros_core.fonts import setup_japanese_font, default_bundled_font_path
from ikaros_core.model import (
    build_sections, init_game, execute_section, score_game,
    alpha_gamma_deg, comm_ok, power_gen,
    GameState,
)
from ikaros_core.plots import plot_bplane, plot_orbits_2d_nominal, plot_beta_maps, geometry_3d_figure


# -----------------------------
# 画面設定
# -----------------------------
st.set_page_config(page_title="IKAROS B-plane Darts (Angle Model)", layout="wide")
st.title("🎯 IKAROS：B-plane ダーツ（角度モデル版）")
st.caption("通信・発電は『帆法線 n と、太陽方向 s・地球方向 e のなす角（α,γ）』だけで決めます。")


cfg = GameConfig()
sections = build_sections()

# フォント（同梱）
font_name, font_path = setup_japanese_font(default_bundled_font_path())


# -----------------------------
# サイドバー（設定＋説明）
# -----------------------------
with st.sidebar:
    st.header("設定")
    seed = st.number_input("シード（同じ問題を再現）", min_value=1, max_value=999999, value=42, step=1)
    show_truth = st.toggle("先生モード：真値を表示", value=False)

    st.divider()
    st.subheader("日本語フォント")
    if font_name:
        st.caption(f"同梱フォントを使用：{font_name}")
    else:
        st.warning("同梱フォントが読めず、日本語が□になる可能性があります。")  # なるべく避けたい…！

    st.divider()
    st.subheader("学習ポイント（角度モデル）")
    st.markdown(
        """
- **発電**：α = angle(n, 太陽方向 s) が小さいほど↑（cos）  
- **通信**：γ = angle(n, 地球方向 e) が小さいほどOK（コーン）  
- つまり **“太陽を向く” vs “地球を向く”** のトレードオフ  
- SRPは弱いので、B-planeは“調整ゲーム”  
"""
    )


# -----------------------------
# セッション状態
# -----------------------------
seed_int = int(seed)
STATE_KEY = "bplane_state_angle_v1"
SEED_KEY = "bplane_seed_angle_v1"
PAGE_KEY = "page_angle_v1"

if STATE_KEY not in st.session_state or st.session_state.get(SEED_KEY) != seed_int:
    st.session_state[STATE_KEY] = init_game(cfg, sections, seed=seed_int)
    st.session_state[SEED_KEY] = seed_int
    st.session_state[PAGE_KEY] = "Play"

state: GameState = st.session_state[STATE_KEY]


def rerun():
    (st.rerun() if hasattr(st, "rerun") else st.experimental_rerun())


def reset():
    st.session_state[STATE_KEY] = init_game(cfg, sections, seed=seed_int)
    st.session_state[PAGE_KEY] = "Play"
    rerun()


if state.phase == "result":
    st.session_state[PAGE_KEY] = "Result"

page = st.radio("ページ", ["Play", "Result"], horizontal=True, index=(0 if st.session_state[PAGE_KEY] == "Play" else 1))
st.session_state[PAGE_KEY] = page


# -----------------------------
# Play
# -----------------------------
def render_play():
    sec = sections[min(state.k, len(sections) - 1)]

    # 現在の角度（α,γ）と通信判定
    alpha, gamma = alpha_gamma_deg(state.beta_in, state.beta_out, state, cfg, sections)
    ok = comm_ok(state.beta_in, state.beta_out, state, cfg, sections)
    Pgen, _, _ = power_gen(state.beta_in, state.beta_out, state, cfg, sections)

    st.progress(min(1.0, state.k / len(sections)))
    st.write(f"進捗：**{state.k}/{len(sections)}** セクション完了（全{len(sections)}）  |  現在：**{sec.name}**（t≈{sec.t_day:.0f}日）")


    # 進めるボタンは上側に置く（操作の主役なので）
    a1, a2, a3, a4, a5, a6 = st.columns([1.0, 1.0, 1.0, 1.0, 1.2, 1.5])
    with a1:
        st.metric("通信", "🟢OK" if ok else "🔴NG")
    with a2:
        st.metric("α（太陽）", f"{alpha:.1f}°")
    with a3:
        st.metric("γ（地球）", f"{gamma:.1f}°")
    with a4:
        st.metric("発電Pgen", f"{Pgen:.1f}")
    with a5:
        st.metric("バッテリ", f"{state.energy:.0f}/{cfg.energy_max:.0f}")
    with a6:
        btn_next = st.button("▶ このセクションを実行（進める）", use_container_width=True, disabled=(state.phase == "result"))
        btn_reset = st.button("🔁 リセット", use_container_width=True)

    if btn_reset:
        reset()
    if btn_next:
        execute_section(state, cfg, sections)
        rerun()

    st.subheader("B-plane（メイン）")
    st.pyplot(plot_bplane(state, cfg, sections, show_truth=show_truth), use_container_width=True)

    # NO-LINKの意味を明確化
    if not sec.uplink_possible:
        st.error("このセクションは NO-LINK：操作できない（Δβ=0固定）。通信もNG扱い。")  # 演出としてのブラックアウト
    else:
        if ok:
            st.success("通信OK：DL可能（中心ほどDL↑）。通信コストも乗ります。")
        else:
            st.warning("通信NG：DLできません（通信コストなし）。")


    left, right = st.columns([1.0, 1.0], gap="large")


    with left:
        st.subheader("位置関係（2D軌道図：ノミナル）")
        st.pyplot(plot_orbits_2d_nominal(state, cfg, sections), use_container_width=True)

        # ライブ推移（変化しない現象を避けるため、軸を明示）
        if state.log:
            df = pd.DataFrame(state.log)
            st.subheader("ライブ推移（主要）")
            st.caption("距離は『ターゲットからどれだけズレているか』。α/γは『太陽/地球との角度』です。")
            st.line_chart(df.set_index("turn")[["dist_to_target_km"]], height=170)
            st.line_chart(df.set_index("turn")[["energy", "alpha_sun_deg", "gamma_earth_deg"]], height=220)


    with right:
        st.subheader("βin×βout マップ（角度モデル）")
        st.pyplot(plot_beta_maps(state, cfg, sections), use_container_width=True)

        st.subheader("幾何（3D表示）")
        st.caption("太陽方向 s / 地球方向 e / 帆法線 n を同時表示（ドラッグで回転）。")
        st.plotly_chart(geometry_3d_figure(state, cfg, sections), use_container_width=True)

        st.subheader("コマンド（βin / βout）")
        cA, cB = st.columns(2)
        with cA:
            bi = st.slider("βin [deg]", -35.0, 35.0, float(state.beta_in), 1.0)
        with cB:
            bo = st.slider("βout [deg]", -35.0, 35.0, float(state.beta_out), 1.0)

        state.beta_in = float(bi)
        state.beta_out = float(bo)


    if state.log:
        with st.expander("ログ（必要なら開く）", expanded=False):
            st.dataframe(pd.DataFrame(state.log), use_container_width=True, hide_index=True)


# -----------------------------
# Result
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
        st.line_chart(df.set_index("turn")[["dist_to_target_km", "energy", "alpha_sun_deg", "gamma_earth_deg", "data_buffer", "data_lost_total"]], height=300)

    if st.button("🔁 もう一回（リセット）", use_container_width=True):
        reset()


# -----------------------------
# ルーティング
# -----------------------------
if page == "Play":
    render_play()
else:
    render_result()
