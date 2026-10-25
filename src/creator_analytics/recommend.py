"""Natural-language recommendations from global SHAP feature explanations.

Public API
----------
build_shap_context(author_id, creator_shap_rows, global_importance,
                   n_train, n_test, size_bin)
    -> ShapContext

get_recommendations(ctx, use_llm=True) -> list[str]
    Returns 2–4 recommendation strings in Ukrainian.
    Falls back to templates if ANTHROPIC_API_KEY is not set or anthropic is unavailable.

Note: SHAP data is optional. If creator_shap_rows is empty / None, top_features
      will be empty and the LLM/templates will work from global_top_features only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from creator_analytics.features import describe_feature

_STABILITY_FALLBACK_N_TRAIN = 150


@dataclass
class ShapContext:
    """Summary of a creator's feature profile derived from the global model's SHAP values."""

    author_id: str
    n_train: int
    n_test: int
    size_bin: str
    # Top features for this creator's test videos (global model SHAP, filtered to creator)
    top_features: list[dict]
    # Global top-5 feature names (averaged across all test videos)
    global_top_features: list[str]
    overlap_with_global: int
    is_stable: bool  # False when n_train < threshold


def build_shap_context(
    author_id: str,
    creator_shap_rows: pd.DataFrame | None,
    global_importance: pd.DataFrame,
    n_train: int,
    n_test: int,
    size_bin: str,
    top_n: int = 5,
) -> ShapContext:
    """Build a ShapContext for one creator using global-model SHAP values.

    Parameters
    ----------
    creator_shap_rows:
        Rows of shap_global_test.parquet for this creator's test videos.
        Each column is a feature, each row a video.  May be None or empty
        if SHAP has not been computed yet — the context is still valid but
        top_features will be empty.
    global_importance:
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending,
        computed over all test videos.
    """
    top_feats: list[dict] = []

    if creator_shap_rows is not None and len(creator_shap_rows) > 0:
        creator_mean_abs = creator_shap_rows.abs().mean()
        creator_mean_signed = creator_shap_rows.mean()
        top_series = creator_mean_abs.sort_values(ascending=False).head(top_n)
        for fname, mean_abs in top_series.items():
            signed = float(creator_mean_signed.get(fname, 0.0))
            direction: str | None = "+" if signed > 0 else "-"
            raw_label = describe_feature(str(fname))
            label = raw_label[:60] if len(raw_label) > 60 else raw_label
            top_feats.append(
                {
                    "feature": str(fname),
                    "label": label,
                    "mean_abs_shap": float(mean_abs),
                    "direction": direction,
                }
            )

    global_top5 = global_importance.head(5)["feature"].tolist()
    creator_top_names = [f["feature"] for f in top_feats]
    overlap = len(set(creator_top_names) & set(global_top5))

    return ShapContext(
        author_id=author_id,
        n_train=n_train,
        n_test=n_test,
        size_bin=size_bin,
        top_features=top_feats,
        global_top_features=global_top5,
        overlap_with_global=overlap,
        is_stable=n_train >= _STABILITY_FALLBACK_N_TRAIN,
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
3. Якщо результатів мало — скажи про це прямо.
4. Рекомендації мають бути конкретними й дієвими, без загальних слів.
5. Відповідай виключно українською мовою.
"""


def _build_llm_prompt(ctx: ShapContext) -> str:
    lines = [
        "Статистика креатора:",
        f"  Навчальних відео: {ctx.n_train}  |  Група розміру: {ctx.size_bin}",
    ]

    if not ctx.is_stable:
        lines.append(
            f"\n⚠️ НЕСТАБІЛЬНІСТЬ: менше {_STABILITY_FALLBACK_N_TRAIN} навч. відео. "
            "Зроби застереження про ненадійність у кожній рекомендації."
        )

    if ctx.top_features:
        lines.append(
            "\nНайважливіші ознаки цього креатора "
            f"(з глобальної моделі, топ-{len(ctx.top_features)}):"
        )
        for i, f in enumerate(ctx.top_features, 1):
            dir_str = ""
            if f["direction"] == "+":
                dir_str = " → вищі значення асоціюються з кращим результатом"
            elif f["direction"] == "-":
                dir_str = " → нижчі значення асоціюються з кращим результатом"
            lines.append(f"  {i}. {f['feature']} («{f['label']}»){dir_str}")
    else:
        lines.append(
            "\nДані SHAP для окремих відео ще не пораховані. Використай глобальні топ-5 ознак."
        )

    global_str = ", ".join(ctx.global_top_features)
    lines.append(f"\nГлобальні топ-5 ознак: {global_str}")
    if ctx.top_features:
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
        import anthropic
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
    import re

    items = [s.strip() for s in re.split(r"\n{2,}|\n(?=\d+[\.\)])", text) if s.strip()]
    return items[:4] if items else [text]


# ------------------------------------------------------------------ #
# Template fallback                                                    #
# ------------------------------------------------------------------ #


def _fallback_recommendations(ctx: ShapContext) -> list[str]:
    recs: list[str] = []

    features_to_use = (
        ctx.top_features
        if ctx.top_features
        else [
            {"feature": f, "label": describe_feature(f), "direction": None}
            for f in ctx.global_top_features[:3]
        ]
    )

    for feat in features_to_use[:3]:
        label = feat["label"]
        if feat.get("direction") == "+":
            recs.append(
                f"Вищі значення показника «{label}» асоціюються з кращими результатами "
                f"у твоїх відео — зверни на нього увагу при плануванні контенту."
            )
        elif feat.get("direction") == "-":
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
            pass
    return _fallback_recommendations(ctx)
