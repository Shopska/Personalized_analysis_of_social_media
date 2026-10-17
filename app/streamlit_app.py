"""Interactive demo (in progress): pick a creator, see predicted performance and why.

Run with ``streamlit run app/streamlit_app.py``. For now it only shows what the demo
will contain; the real version comes once the per-creator models and SHAP explanations
are ready.
"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Creator performance analytics", page_icon="\U0001f4ca")
    st.title("\U0001f4ca Creator performance analytics")
    st.info("Demo in progress — models are not trained yet.")
    st.markdown(
        "Planned:\n"
        "- choose a creator from the dataset\n"
        "- see the predicted performance of their videos (per-creator vs. global model)\n"
        "- see which features drive the prediction for this creator (SHAP)\n"
    )


if __name__ == "__main__":
    main()
