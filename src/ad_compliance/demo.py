"""Small Streamlit demo for checking a piece of ad copy."""

from __future__ import annotations

import streamlit as st

from ad_compliance.retrieve import format_debug_report, retrieve
from ad_compliance.verdict import check_ad, format_report

DISCLAIMER = (
    "This is a technical demonstration, not legal advice. The index is built from "
    "CAP/BCAP Advertising Guidance on protecting under-18s (2022). That document "
    "is not a Gambling Commission licence condition and does not bind the ASA."
)

EXAMPLES = {
    "Clear concern (cartoon Santa)": (
        "New cartoon slots! Help Santa and his cuddly elf pals open magical loot "
        "sacks. Spin now — it's as easy as a Christmas cartoon!"
    ),
    "Likely ok (age-verified email)": (
        "Email to a logged-in, age-verified 18+ customer list: 'This weekend's "
        "racing card at Newbury is up. 18+ only. Please gamble responsibly.' "
        "Plain typography, no imagery."
    ),
    "Borderline (retired footballer pundit)": (
        "A well-known retired Premier League striker, now a weekly pundit, "
        "smiles into camera and says the weekend's fixtures are up on our app."
    ),
}


def render() -> None:
    st.set_page_config(page_title="Ad Compliance Checker", layout="wide")
    st.title("Ad Compliance Checker")
    st.caption("UK gambling advertising × CAP/BCAP under-18s guidance")
    st.info(DISCLAIMER)

    example = st.selectbox("Start from an example (optional)", ["(blank)", *EXAMPLES])
    default = "" if example == "(blank)" else EXAMPLES[example]
    ad_text = st.text_area("Advertisement copy", value=default, height=180)

    col_check, col_retrieve = st.columns(2)
    run_check = col_check.button("Run check", type="primary")
    run_retrieve = col_retrieve.button("Debug retrieve only")

    if run_check:
        if not ad_text.strip():
            st.error("Paste some ad copy first.")
            return
        with st.spinner("Retrieving guidance and judging each section…"):
            try:
                report = check_ad(ad_text)
            except Exception as exc:
                st.error(str(exc))
                st.caption(
                    "Ingest the PDF first (`ad-compliance ingest`) and set an LLM provider key."
                )
                return
        _colour = {"breach": "🔴", "needs_review": "🟠", "ok": "🟢"}
        st.subheader(f"{_colour.get(report.overall, '')} Overall: {report.overall}")
        st.write(report.summary)
        st.code(format_report(report), language="text")
        st.json(report.model_dump())

    if run_retrieve:
        if not ad_text.strip():
            st.error("Paste some ad copy first.")
            return
        with st.spinner("Retrieving…"):
            try:
                hits = retrieve(ad_text)
            except Exception as exc:
                st.error(str(exc))
                return
        st.code(format_debug_report(ad_text, hits), language="text")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
