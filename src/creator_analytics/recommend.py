"""Natural-language recommendations from SHAP feature explanations.

Public API
----------
build_shap_context(author_id, pc_shap_df, global_importance, pc_meta,
                   rank_corr_df, stability_df=None, signed_shap_per_feature=None)
    -> ShapContext

get_recommendations(ctx, use_llm=True) -> list[str]
    Returns 2–4 recommendation strings in Ukrainian.
    Falls back to templates if ANTHROPIC_API_KEY is not set or anthropic is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from creator_analytics.features import describe_feature

_STABILITY_THRESHOLD = 0.4
_STABILITY_FALLBACK_N_TRAIN = 150


@dataclass
class ShapContext:
    """Summary of a creator's SHAP-derived feature profile."""

    author_id: str
    n_train: int
    n_test: int
    size_bin: str
    # Each dict: feature, label, mean_abs_shap, direction ("+"/"-"/None)
    top_features: list[dict]
    global_top_features: list[str]
    rank_corr_global: float
    stability_score: float | None
    is_stable: bool
    overlap_with_global: int


def build_shap_context(
    author_id: str,
    pc_shap_df: pd.DataFrame,
    global_importance: pd.DataFrame,
    pc_meta: pd.DataFrame,
    rank_corr_df: pd.DataFrame,
    stability_df: pd.DataFrame | None = None,
    signed_shap_per_feature: dict[str, float] | None = None,
    top_n: int = 5,
) -> ShapContext:
    """Assemble a ShapContext for one creator from pre-computed SHAP data.

    Parameters
    ----------
    signed_shap_per_feature:
        Optional dict mapping feature name -> average signed SHAP value for
        this creator (computed externally from the creator's RF model).
        Used to determine direction ("+"/"-") in recommendations.
    """
    meta_row = pc_meta[pc_meta["author_id"] == author_id]
    if meta_row.empty:
        raise ValueError(f"Creator {author_id!r} not found in pc_meta")
    meta = meta_row.iloc[0]

    creator_shap = (
        pc_shap_df[pc_shap_df["author_id"] == author_id]
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )

    top_feats: list[dict] = []
    for _, row in creator_shap.iterrows():
        fname = row["feature"]
        raw_label = describe_feature(fname)
        label = raw_label[:60] if len(raw_label) > 60 else raw_label
        direction: str | None = None
        if signed_shap_per_feature and fname in signed_shap_per_feature:
            v = signed_shap_per_feature[fname]
            direction = "+" if v > 0 else "-"
        top_feats.append({
            "feature": fname,
            "label": label,
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "direction": direction,
        })

    global_top5 = global_importance.head(5)["feature"].tolist()

    rho_row = rank_corr_df[rank_corr_df["author_id"] == author_id]
    rho = float(rho_row["spearman_vs_global"].values[0]) if not rho_row.empty else float("nan")

    stability: float | None = None
    if stability_df is not None and not stability_df.empty:
        stab_row = stability_df[stability_df["author_id"] == author_id]
        if not stab_row.empty:
            stability = float(stab_row["stability"].values[0])

    creator_top_names = [f["feature"] for f in top_feats]
    overlap = len(set(creator_top_names) & set(global_top5))

    if stability is not None:
        is_stable = stability >= _STABILITY_THRESHOLD
    else:
        is_stable = int(meta["n_train"]) >= _STABILITY_FALLBACK_N_TRAIN

    return ShapContext(
        author_id=author_id,
        n_train=int(meta["n_train"]),
        n_test=int(meta["n_test"]),
        size_bin=str(meta["size_bin"]),
        top_features=top_feats,
        global_top_features=global_top5,
        rank_corr_global=rho,
        stability_score=stability,
        is_stable=is_stable,
        overlap_with_global=overlap,
    )


# ------------------------------------------------------------------ #
# LLM-based recommendations                                           #
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
Ти — аналітик даних, який допомагає TikTok-креатору зрозуміти свою контент-стратегію \
на основі аналізу ML-моделі.

Обов'язкові правила:
1. Ніколи не обіцяй конкретну кількість переглядів, підписників або охоплення.
2. Використовуй «асоціюється з», а не «призводить до» чи «гарантує».
3. Якщо результати нестабільні або даних мало — скажи про це прямо.
4. Рекомендації мають бути конкретними й дієвими, без загальних слів.
5. Відповідай виключно українською мовою.
"""


def _build_llm_prompt(ctx: ShapContext) -> str:
    lines = [
        f"Статистика креатора:",
        f"  Навчальних відео: {ctx.n_train}  |  Група розміру: {ctx.size_bin}",
        f"  Схожість профілю з «середнім» (ρ): {ctx.rank_corr_global:.2f}",
    ]

    if not ctx.is_stable:
        stab_detail = (
            f"{ctx.stability_score:.2f}" if ctx.stability_score is not None
            else f"н/д (менше {_STABILITY_FALLBACK_N_TRAIN} навч. відео)"
        )
        lines.append(
            f"\n⚠️ НЕСТАБІЛЬНІСТЬ: стабільність топ-3 ознак = {stab_detail}. "
            "Зроби застереження про ненадійність у кожній рекомендації."
        )

    lines.append(f"\nНайважливіші ознаки цього креатора (топ-{len(ctx.top_features)}):")
    for i, f in enumerate(ctx.top_features, 1):
        dir_str = ""
        if f["direction"] == "+":
            dir_str = " → вищі значення асоціюються з кращим результатом"
        elif f["direction"] == "-":
            dir_str = " → нижчі значення асоціюються з кращим результатом"
        lines.append(f"  {i}. {f['feature']} («{f['label']}»){dir_str}")

    global_str = ", ".join(ctx.global_top_features)
    lines.append(f"\nГлобальні топ-5 ознак (для порівняння): {global_str}")
    lines.append(
        f"Збіг з глобальними топ-5: {ctx.overlap_with_global}/5 "
        f"({'унікальний профіль' if ctx.overlap_with_global <= 1 else 'схожий на середній'})"
    )
    lines.append(
        "\nДай 2–4 конкретні рекомендації. "
        "Формат: одне речення-суть + одне речення-обґрунтування. "
        "Без кліше про «автентичність» та «регулярність»."
    )
    return "\n".join(lines)


def _call_llm(ctx: ShapContext) -> list[str]:
    try:
        import anthropic  # lazy import — optional dependency
    except ImportError:
        raise RuntimeError("anthropic package not installed")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_llm_prompt(ctx)}],
    )
    text = response.content[0].text.strip()
    # Split on blank lines or numbered items; return as list
    import re
    items = [s.strip() for s in re.split(r"\n{2,}|\n(?=\d+[\.\)])", text) if s.strip()]
    return items[:4] if items else [text]


# ------------------------------------------------------------------ #
# Template fallback                                                    #
# ------------------------------------------------------------------ #

def _fallback_recommendations(ctx: ShapContext) -> list[str]:
    recs: list[str] = []

    for feat in ctx.top_features[:3]:
        label = feat["label"]
        if feat["direction"] == "+":
            recs.append(
                f"Вищі значення показника «{label}» асоціюються з кращими результатами "
                f"у твоїх відео — зверни на нього увагу при плануванні контенту."
            )
        elif feat["direction"] == "-":
            recs.append(
                f"Нижчі значення показника «{label}» асоціюються з кращими результатами "
                f"у твоїх відео — розглянь, як це можна оптимізувати."
            )
        else:
            recs.append(
                f"Показник «{label}» є одним з найважливіших для твоїх відео — "
                f"проаналізуй, як він змінюється між твоїми найуспішнішими і найменш "
                f"успішними публікаціями."
            )

    if not ctx.is_stable:
        recs.append(
            f"⚠️ У тебе {ctx.n_train} навчальних відео — цього недостатньо для стабільних "
            f"інсайтів. Ці закономірності можуть змінитися з накопиченням більшої кількості "
            f"відео; не роби різких змін стратегії на основі лише цих даних."
        )

    return recs[:4]


# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #

def get_recommendations(ctx: ShapContext, use_llm: bool = True) -> list[str]:
    """Return 2–4 Ukrainian recommendation strings for a creator.

    Uses Claude API if ``use_llm=True`` and ``ANTHROPIC_API_KEY`` is set;
    otherwise falls back to deterministic templates.
    """
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _call_llm(ctx)
        except Exception:
            pass  # silently fall back to templates
    return _fallback_recommendations(ctx)
