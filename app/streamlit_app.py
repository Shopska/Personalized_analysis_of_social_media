"""Creator analytics demo: pick a creator, explore their SHAP profile and AI recommendations.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from creator_analytics.features import FEATURE_DESCRIPTIONS  # noqa: E402
from creator_analytics.recommend import build_shap_context, get_recommendations  # noqa: E402

# ------------------------------------------------------------------ #
# Required result files                                               #
# ------------------------------------------------------------------ #
_REQUIRED = [
    ROOT / "results" / "shap_per_creator_importance.parquet",
    ROOT / "results" / "shap_rank_corr.parquet",
    ROOT / "results" / "per_creator_metrics_percreator.parquet",
    ROOT / "results" / "predictions_per_creator.parquet",
    ROOT / "models" / "global_xgb.joblib",
]


# ------------------------------------------------------------------ #
# Cached loaders                                                      #
# ------------------------------------------------------------------ #

@st.cache_data
def load_app_data(root: str) -> dict:
    r = Path(root)
    pc_shap    = pd.read_parquet(r / "results" / "shap_per_creator_importance.parquet")
    rank_corr  = pd.read_parquet(r / "results" / "shap_rank_corr.parquet")
    pc_meta    = pd.read_parquet(r / "results" / "per_creator_metrics_percreator.parquet")
    preds_pc   = pd.read_parquet(r / "results" / "predictions_per_creator.parquet")

    stab_path  = r / "results" / "shap_stability.parquet"
    stability  = pd.read_parquet(stab_path) if stab_path.exists() else pd.DataFrame()

    global_shap_path = r / "results" / "shap_global_test.parquet"
    global_shap = pd.read_parquet(global_shap_path) if global_shap_path.exists() else None

    # Pre-compute global importance
    if global_shap is not None:
        mean_abs = global_shap.abs().mean()
        global_importance = (
            mean_abs.reset_index()
            .rename(columns={"index": "feature", 0: "mean_abs_shap"})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
    else:
        global_importance = pd.DataFrame(columns=["feature", "mean_abs_shap"])

    return {
        "pc_shap":          pc_shap,
        "rank_corr":        rank_corr,
        "pc_meta":          pc_meta,
        "preds_pc":         preds_pc,
        "stability":        stability,
        "global_importance": global_importance,
    }


@st.cache_data(show_spinner="Завантажую дані та будую тест-спліт…")
def load_test_split(root: str):
    from creator_analytics.config import load_config
    from creator_analytics.data import filter_creators, load_raw_table, train_test_split_by_creator
    from creator_analytics.models import prepare_X
    from creator_analytics.targets import build_target, get_nonmonotonic_video_ids

    r      = Path(root)
    config = load_config(str(r / "config" / "default.yaml"))
    RAW    = r / config["paths"]["raw_dir"]
    seed   = config["seed"]

    videos        = load_raw_table(str(RAW), "videos")
    creator_daily = load_raw_table(str(RAW), "creator_daily")
    engagement    = load_raw_table(str(RAW), "engagement_daily")

    bad_ids      = get_nonmonotonic_video_ids(engagement)
    videos_clean = videos[~videos["video_id"].isin(bad_ids)].reset_index(drop=True)
    videos_f     = filter_creators(
        videos_clean, "author_id",
        min_videos=config["data"]["min_videos_per_creator"],
    )

    train_df, test_df = train_test_split_by_creator(
        videos_f, "author_id", test_size=config["split"]["test_size"], seed=seed
    )
    train_ids = set(train_df["video_id"])
    test_ids  = set(test_df["video_id"])

    feat_path = r / "data" / "processed" / "features.parquet"
    features  = (
        pd.read_parquet(feat_path) if feat_path.exists()
        else __import__("creator_analytics.features", fromlist=["build_features"]).build_features(
            videos_f, creator_daily, train_video_ids=train_ids
        )
    )

    rel_t = build_target(engagement, videos_f, day=14, mode="relative",
                         train_video_ids=train_ids)
    data  = (features
             .merge(videos_f[["video_id", "author_id"]], on="video_id")
             .merge(rel_t, on="video_id"))

    test_mask   = data["video_id"].isin(test_ids)
    X_all       = prepare_X(data.drop(columns=["author_id", "target"]))
    X_test      = X_all[test_mask].reset_index(drop=True)
    y_test      = data.loc[test_mask, "target"].reset_index(drop=True)
    author_test = data.loc[test_mask, "author_id"].reset_index(drop=True)
    video_ids   = data.loc[test_mask, "video_id"].reset_index(drop=True)
    return X_test, y_test, author_test, video_ids


@st.cache_resource
def load_global_model(root: str):
    import joblib
    return joblib.load(Path(root) / "models" / "global_xgb.joblib")


def load_creator_model(author_id: str, root: Path):
    import joblib
    path = root / "models" / "per_creator" / "RF" / f"{author_id}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

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
    ax.set_title("SHAP для цього відео", fontsize=10)
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

    data = load_app_data(str(ROOT))
    pc_shap          = data["pc_shap"]
    rank_corr        = data["rank_corr"]
    pc_meta          = data["pc_meta"]
    preds_pc         = data["preds_pc"]
    stability        = data["stability"]
    global_importance = data["global_importance"]

    # --- Anonymization ---
    all_aids       = sorted(pc_meta["author_id"].unique())
    creator_labels = {aid: f"Creator {i+1:03d}" for i, aid in enumerate(all_aids)}
    label_to_aid   = {v: k for k, v in creator_labels.items()}

    def _display_label(aid: str) -> str:
        row = pc_meta[pc_meta["author_id"] == aid].iloc[0]
        return f"{creator_labels[aid]} ({row['size_bin']}, {row['n_train']} train)"

    # -------- Sidebar --------
    with st.sidebar:
        st.title("📊 Creator Analytics")
        st.caption("Персоналізований аналіз контент-стратегії TikTok-креаторів на основі SHAP.")

        display_options = [_display_label(aid) for aid in all_aids]
        selected_display = st.selectbox("Оберіть креатора:", display_options, index=0)
        selected_label   = selected_display.split(" (")[0]
        author_id        = label_to_aid[selected_label]

        meta_row    = pc_meta[pc_meta["author_id"] == author_id].iloc[0]
        creator_rho = rank_corr[rank_corr["author_id"] == author_id]
        rho_val     = float(creator_rho["spearman_vs_global"].values[0]) if len(creator_rho) else float("nan")

        with st.expander("ℹ️ Про модель"):
            st.markdown(
                "**Що робить модель?**\n\n"
                "Модель прогнозує відносну популярність відео — наскільки воно "
                "отримає більше або менше переглядів, ніж типове відео цього ж "
                "автора.\n\n"
                "Використовується два підходи:\n"
                "- **Глобальна модель** (XGBoost) — навчена на всіх креаторах.\n"
                "- **Персональна модель** (RF) — окремо для кожного креатора, "
                "тільки на його відео.\n\n"
                "SHAP пояснює, які ознаки вплинули на прогноз."
            )

    # -------- Creator SHAP context --------
    ctx = build_shap_context(
        author_id=author_id,
        pc_shap_df=pc_shap,
        global_importance=global_importance,
        pc_meta=pc_meta,
        rank_corr_df=rank_corr,
        stability_df=stability if len(stability) > 0 else None,
    )

    # -------- Tabs --------
    tab1, tab2, tab3 = st.tabs(["📊 SHAP-профіль", "💡 Рекомендації", "🎬 Відео"])

    # ================================================================ #
    # Tab 1 — SHAP profile                                             #
    # ================================================================ #
    with tab1:
        st.subheader(f"{selected_label} — SHAP-профіль")

        creator_shap_top = (
            pc_shap[pc_shap["author_id"] == author_id]
            .sort_values("mean_abs_shap", ascending=False)
            .head(10)
        )
        global_top10 = global_importance.head(10)

        col_l, col_r = st.columns(2)
        with col_l:
            if len(creator_shap_top) > 0:
                fig = _bar_chart(
                    creator_shap_top["feature"].tolist(),
                    creator_shap_top["mean_abs_shap"].tolist(),
                    "#1565C0",
                    f"{selected_label} — топ ознаки",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("SHAP-дані для цього креатора відсутні.")

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

        # Metrics row
        m1, m2, m3 = st.columns(3)
        color_rho = _rho_color(rho_val)
        rho_str   = f"{rho_val:.3f}" if not np.isnan(rho_val) else "n/a"
        m1.metric("Спірмен ρ vs глобальна", rho_str)

        overlap_n = ctx.overlap_with_global
        m2.metric("Збіг топ-5 з глобальною", f"{overlap_n}/5")

        stab_str = (
            f"{ctx.stability_score:.2f}" if ctx.stability_score is not None else "n/a"
        )
        m3.metric("Стабільність топ-3 (bootstrap)", stab_str)

        if not ctx.is_stable:
            if meta_row["n_train"] < 100:
                st.warning(
                    f"⚠️ Цей креатор має лише **{meta_row['n_train']} навчальних відео** — "
                    "замало для надійних персональних закономірностей. "
                    "Розглядайте рекомендації як орієнтовні."
                )
            elif ctx.stability_score is not None and ctx.stability_score < 0.4:
                st.warning(
                    f"⚠️ Стабільність топ-3 ознак: **{ctx.stability_score:.2f}** — "
                    "при перенавчанні на вибірках ті ж самі ознаки з'являються менш ніж у 40% випадків. "
                    "Висновки можуть змінитися з новими даними."
                )

    # ================================================================ #
    # Tab 2 — Recommendations                                          #
    # ================================================================ #
    with tab2:
        st.subheader("💡 Персональні рекомендації")

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
            "Рекомендації згенеровано на основі SHAP-аналізу моделі. "
            "Не є гарантією конкретних результатів."
        )

    # ================================================================ #
    # Tab 3 — Video prediction                                         #
    # ================================================================ #
    with tab3:
        st.subheader("🎬 Прогноз для конкретного відео")

        creator_preds = preds_pc[preds_pc["author_id"] == author_id].reset_index(drop=True)
        if len(creator_preds) == 0:
            st.warning("Прогнози для цього креатора відсутні.")
            st.stop()

        n_videos = len(creator_preds)

        if f"video_idx_{author_id}" not in st.session_state:
            st.session_state[f"video_idx_{author_id}"] = 0

        if st.button("🔀 Інше відео"):
            st.session_state[f"video_idx_{author_id}"] = int(
                np.random.randint(0, n_videos)
            )

        video_idx = st.session_state[f"video_idx_{author_id}"]
        row       = creator_preds.iloc[video_idx]
        y_true    = float(row["y_true"])
        y_pred    = float(row["y_pred"])
        error     = y_pred - y_true

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Фактичний результат (відн.)", f"{y_true:+.3f}")
        vc2.metric("Прогноз моделі",              f"{y_pred:+.3f}")
        vc3.metric("Похибка",                     f"{error:+.3f}")

        st.caption(f"Відео {video_idx + 1} з {n_videos} тест-відео цього креатора.")

        # Per-video SHAP with sign
        cm = load_creator_model(author_id, ROOT)
        if cm is None:
            st.warning("Файл моделі для цього креатора не знайдено.")
        else:
            try:
                with st.spinner("Обчислюю SHAP для цього відео…"):
                    # Load test split (cached after first call)
                    X_test, _, author_test, _ = load_test_split(str(ROOT))

                    mask         = (author_test == author_id).values
                    X_creator    = X_test.iloc[mask][cm.feature_names]
                    n_c          = len(X_creator)
                    vid_i        = video_idx % n_c
                    video_X      = X_creator.iloc[vid_i : vid_i + 1]

                    pre          = cm.model.named_steps["pre"]
                    mdl          = cm.model.named_steps["mdl"]
                    X_enc        = pre.transform(video_X)
                    feat_names   = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
                    explainer    = shap.TreeExplainer(mdl)
                    sv           = explainer.shap_values(X_enc)
                    shap_vals    = sv[0]

                top10_idx  = np.argsort(np.abs(shap_vals))[::-1][:10]
                top10_feats = [feat_names[i] for i in top10_idx]
                top10_vals  = [float(shap_vals[i]) for i in top10_idx]

                fig = _signed_bar_chart(top10_feats, top10_vals)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            except Exception as exc:
                st.warning(f"Не вдалося обчислити SHAP: {exc}")


if __name__ == "__main__":
    main()
