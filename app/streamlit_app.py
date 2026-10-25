"""Creator analytics demo: pick a creator, explore their SHAP profile and recommendations.

Run from the project root:
    streamlit run app/streamlit_app.py

Required:
    models/global_xgb.joblib
    data/processed/dataset_index.parquet

Optional (graceful degradation when missing):
    results/shap_global_test.parquet   → SHAP tabs show info message instead of crashing
    results/predictions_global.parquet → Video tab falls back to live inference
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from creator_analytics.features import FEATURE_DESCRIPTIONS  # noqa: E402
from creator_analytics.recommend import build_shap_context, get_recommendations  # noqa: E402

# ------------------------------------------------------------------ #
# Required result files                                               #
# ------------------------------------------------------------------ #
_REQUIRED = [
    ROOT / "models" / "global_xgb.joblib",
    ROOT / "data" / "processed" / "dataset_index.parquet",
]


# ------------------------------------------------------------------ #
# Cached loaders                                                      #
# ------------------------------------------------------------------ #


@st.cache_data
def load_creator_stats(root: str) -> pd.DataFrame:
    """Load per-creator train/test counts and size_bin from dataset_index."""
    idx = pd.read_parquet(Path(root) / "data" / "processed" / "dataset_index.parquet")
    train_counts = idx[idx["split"] == "train"].groupby("author_id").size().rename("n_train")
    test_counts = idx[idx["split"] == "test"].groupby("author_id").size().rename("n_test")
    stats = (
        train_counts.to_frame().join(test_counts, how="outer").fillna(0).astype(int).reset_index()
    )
    stats["size_bin"] = stats["n_train"].apply(_size_bin)
    return stats


@st.cache_data
def load_shap_data(root: str) -> tuple[pd.DataFrame | None, pd.DataFrame, pd.Series | None]:
    """Load global SHAP matrix, global importance, and test author_ids.

    Returns (shap_df, global_importance, auth_test).
    shap_df and auth_test are None if shap_global_test.parquet is missing.
    """
    r = Path(root)
    shap_path = r / "results" / "shap_global_test.parquet"
    if not shap_path.exists():
        empty_imp = pd.DataFrame(columns=["feature", "mean_abs_shap"])
        return None, empty_imp, None

    shap_df = pd.read_parquet(shap_path)
    mean_abs = shap_df.abs().mean()
    global_importance = (
        mean_abs.reset_index()
        .rename(columns={"index": "feature", 0: "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    idx = pd.read_parquet(r / "data" / "processed" / "dataset_index.parquet")
    auth_test = idx[idx["split"] == "test"]["author_id"].reset_index(drop=True)
    return shap_df, global_importance, auth_test


@st.cache_data
def load_predictions(root: str) -> pd.DataFrame | None:
    path = Path(root) / "results" / "predictions_global.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource
def load_global_model(root: str):
    import joblib

    return joblib.load(Path(root) / "models" / "global_xgb.joblib")


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _size_bin(n: int) -> str:
    if n < 50:
        return "micro (<50)"
    if n < 100:
        return "small (50–100)"
    if n < 300:
        return "medium (100–300)"
    return "large (300+)"


def _short(fname: str, n: int = 36) -> str:
    desc = FEATURE_DESCRIPTIONS.get(fname, fname)
    return desc[:n] if len(desc) > n else desc


def _rho_color(rho: float) -> str:
    if rho > 0.5:
        return "green"
    if rho > 0.3:
        return "orange"
    return "red"


def _bar_chart(features: list[str], values: list[float], color: str, title: str):
    labels = [_short(f) for f in features]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(labels[::-1], values[::-1], color=color)
    ax.set_xlabel("Mean |SHAP|", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _signed_bar_chart(feat_names: list[str], shap_vals: list[float]):
    labels = [_short(f) for f in feat_names]
    colors = ["#2E7D32" if v >= 0 else "#C62828" for v in shap_vals]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(labels[::-1], shap_vals[::-1], color=colors[::-1])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP value (+ = підвищує прогноз)", fontsize=9)
    ax.set_title("SHAP для цього відео (глобальна модель)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Main app                                                            #
# ------------------------------------------------------------------ #


def main() -> None:
    st.set_page_config(
        page_title="Creator Analytics",
        page_icon="📊",
        layout="wide",
    )

    # --- Guard: required files ---
    missing = [p for p in _REQUIRED if not p.exists()]
    if missing:
        st.error(
            "Моделі ще не навчені або результати відсутні. "
            "Запустіть: **`make reproduce`**\n\n"
            f"Відсутні файли: {[str(p.relative_to(ROOT)) for p in missing]}"
        )
        st.stop()

    creator_stats = load_creator_stats(str(ROOT))
    shap_df, global_importance, auth_test = load_shap_data(str(ROOT))
    preds_global = load_predictions(str(ROOT))
    shap_available = shap_df is not None

    all_aids = sorted(creator_stats["author_id"].unique())
    creator_labels = {aid: f"Creator {i+1:03d}" for i, aid in enumerate(all_aids)}
    label_to_aid = {v: k for k, v in creator_labels.items()}

    def _display_label(aid: str) -> str:
        row = creator_stats[creator_stats["author_id"] == aid].iloc[0]
        return f"{creator_labels[aid]} ({row['size_bin']}, {row['n_train']} train)"

    # -------- Sidebar --------
    with st.sidebar:
        st.title("📊 Creator Analytics")
        st.caption("Персоналізований аналіз контент-стратегії TikTok-креаторів на основі SHAP.")

        display_options = [_display_label(aid) for aid in all_aids]
        selected_display = st.selectbox("Оберіть креатора:", display_options, index=0)
        selected_label = selected_display.split(" (")[0]
        author_id = label_to_aid[selected_label]

        meta_row = creator_stats[creator_stats["author_id"] == author_id].iloc[0]

        with st.expander("ℹ️ Про модель"):
            st.markdown(
                "**Що робить модель?**\n\n"
                "Модель прогнозує відносну популярність відео — наскільки воно "
                "отримає більше або менше переглядів, ніж типове відео цього ж "
                "автора.\n\n"
                "Використовується **глобальна XGBoost-модель**, навчена на всіх "
                "креаторах одночасно. SHAP пояснює, які ознаки вплинули на прогноз."
            )

        if not shap_available:
            st.info(
                "ℹ️ SHAP-дані ще не пораховані.\n\n"
                "Щоб увімкнути вкладки SHAP і Рекомендації, виконайте NB07."
            )

    # -------- SHAP context (works without SHAP files) --------
    n_train = int(meta_row["n_train"])
    n_test = int(meta_row["n_test"])
    size_bin = str(meta_row["size_bin"])

    if shap_available and auth_test is not None:
        creator_mask = auth_test == author_id
        creator_shap_rows = shap_df[creator_mask.values]
    else:
        creator_shap_rows = None

    ctx = build_shap_context(
        author_id=author_id,
        creator_shap_rows=creator_shap_rows,
        global_importance=global_importance,
        n_train=n_train,
        n_test=n_test,
        size_bin=size_bin,
    )

    # -------- Tabs --------
    tab1, tab2, tab3 = st.tabs(["📊 SHAP-профіль", "💡 Рекомендації", "🎬 Відео"])

    # ================================================================ #
    # Tab 1 — SHAP profile                                             #
    # ================================================================ #
    with tab1:
        st.subheader(f"{selected_label} — SHAP-профіль (глобальна модель)")

        if not shap_available:
            st.info(
                "SHAP-дані ще не пораховані. "
                "Запустіть NB07 (або відповідний скрипт), щоб побачити SHAP-профіль."
            )
        else:
            col_l, col_r = st.columns(2)

            creator_top10 = (
                shap_df[auth_test == author_id].abs().mean().sort_values(ascending=False).head(10)
            )
            with col_l:
                if len(creator_top10) > 0:
                    fig = _bar_chart(
                        creator_top10.index.tolist(),
                        creator_top10.values.tolist(),
                        "#1565C0",
                        f"{selected_label} — топ ознаки",
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            global_top10 = global_importance.head(10)
            with col_r:
                if len(global_top10) > 0:
                    fig = _bar_chart(
                        global_top10["feature"].tolist(),
                        global_top10["mean_abs_shap"].tolist(),
                        "#E53935",
                        "Глобальна модель — топ ознаки",
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            m1, m2 = st.columns(2)
            m1.metric("Навчальних відео", n_train)
            m2.metric("Збіг топ-5 з глобальною", f"{ctx.overlap_with_global}/5")

            if not ctx.is_stable:
                st.warning(
                    f"⚠️ Цей креатор має лише **{n_train} навчальних відео** — "
                    "замало для надійних персональних закономірностей. "
                    "Розглядайте рекомендації як орієнтовні."
                )

    # ================================================================ #
    # Tab 2 — Recommendations                                          #
    # ================================================================ #
    with tab2:
        st.subheader("💡 Персональні рекомендації")

        if not shap_available and not global_importance.empty:
            st.info(
                "SHAP розраховано тільки на глобальному рівні. "
                "Рекомендації базуються на глобальних топ-ознаках."
            )
        elif not shap_available:
            st.info(
                "SHAP-дані ще не пораховані. "
                "Запустіть NB07, щоб увімкнути персональні рекомендації."
            )

        regen_key = f"recs_{author_id}"
        if regen_key not in st.session_state or st.button("🔄 Оновити рекомендації"):
            with st.spinner("Генерую рекомендації…"):
                recs = get_recommendations(ctx, use_llm=True)
            st.session_state[regen_key] = recs

        recs = st.session_state.get(regen_key, [])
        if recs:
            for i, rec in enumerate(recs, 1):
                st.markdown(
                    f"<div style='background:#f0f4ff;padding:12px 16px;"
                    f"border-left:4px solid #1565C0;border-radius:4px;margin:8px 0'>"
                    f"<b>{i}.</b> {rec}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Рекомендації недоступні.")

        st.caption(
            "Рекомендації базуються на SHAP-аналізі глобальної моделі. "
            "Не є гарантією конкретних результатів."
        )

    # ================================================================ #
    # Tab 3 — Video predictions                                         #
    # ================================================================ #
    with tab3:
        st.subheader("🎬 Прогноз для конкретного відео")

        if preds_global is not None:
            creator_preds = preds_global[preds_global["author_id"] == author_id].reset_index(
                drop=True
            )
        else:
            creator_preds = pd.DataFrame()

        if len(creator_preds) == 0:
            st.info(
                "Прогнози для цього креатора відсутні. "
                "Запустіть NB04 або NB07 для генерації predictions_global.parquet."
            )
        else:
            n_videos = len(creator_preds)
            if f"video_idx_{author_id}" not in st.session_state:
                st.session_state[f"video_idx_{author_id}"] = 0

            if st.button("🔀 Інше відео"):
                st.session_state[f"video_idx_{author_id}"] = int(np.random.randint(0, n_videos))

            video_idx = st.session_state[f"video_idx_{author_id}"]
            row = creator_preds.iloc[video_idx]
            y_true = float(row["y_true"])
            y_pred = float(row["y_pred"])
            error = y_pred - y_true

            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Фактичний результат (відн.)", f"{y_true:+.3f}")
            vc2.metric("Прогноз (глобальна модель)", f"{y_pred:+.3f}")
            vc3.metric("Похибка", f"{error:+.3f}")

            st.caption(f"Відео {video_idx + 1} з {n_videos} тест-відео цього креатора.")

            # Per-video SHAP from global model
            if not shap_available:
                st.info("SHAP для окремих відео з'явиться після запуску NB07.")
            else:
                try:
                    creator_shap_vids = shap_df[(auth_test == author_id).values]
                    if len(creator_shap_vids) > 0:
                        vid_i = video_idx % len(creator_shap_vids)
                        shap_row = creator_shap_vids.iloc[vid_i]
                        top10_idx = shap_row.abs().sort_values(ascending=False).head(10).index
                        fig = _signed_bar_chart(
                            list(top10_idx),
                            [float(shap_row[f]) for f in top10_idx],
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                except Exception as exc:
                    st.warning(f"Не вдалося відобразити SHAP: {exc}")


if __name__ == "__main__":
    main()
