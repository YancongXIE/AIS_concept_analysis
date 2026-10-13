"""Streamlit interface for AIS Ngram.

Local use:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

import ais_ngram
from app import DEFAULT_TERMS, _caption, _chart_title, run_query

st.set_page_config(page_title="AIS Ngram", layout="wide")

FIELD_OPTIONS = ["title", "abstract", "keywords"]
TYPE_OPTIONS = ["Conference", "Journal", "Other"]
DENOMINATORS = {
    "Word share: occurrences / all words": "tokens",
    "Paper share: papers with text that contain the term": "with_text",
    "Paper share: all papers that contain the term": "all",
}


@st.cache_resource(show_spinner="Loading the corpus...")
def load_corpus():
    return ais_ngram.load_corpus()


def main() -> None:
    corpus = load_corpus()
    year_lo = int(corpus["year"].min())
    year_hi = int(corpus["year"].max())

    st.title("AIS Ngram")
    st.caption(
        "Share of a word or phrase in titles, abstracts, and keywords from "
        "aisel_all_papers_20260928. The default matches Google Ngram: "
        "occurrences divided by total words."
    )

    with st.sidebar:
        st.header("Filters")
        year_min, year_max = st.slider(
            "Years",
            min_value=year_lo,
            max_value=year_hi,
            value=(year_lo, year_hi),
        )
        types = st.multiselect("Document type", TYPE_OPTIONS, default=TYPE_OPTIONS)
        fields = st.multiselect("Fields", FIELD_OPTIONS, default=FIELD_OPTIONS)
        venue = st.text_input("Venue contains", placeholder="Optional")
        denominator_label = st.radio("Percentage", list(DENOMINATORS), index=0)
        denominator = DENOMINATORS[denominator_label]
        run = st.button("Calculate and plot", type="primary")

    st.markdown("**Words or phrases**")
    st.caption("Synonyms: Sociotechnical = socio-technical, socio technical")
    terms_text = st.text_area(
        "Words or phrases",
        value=DEFAULT_TERMS,
        height=260,
        label_visibility="collapsed",
        help=(
            "One term per line. Matching is case-insensitive and whole-word: Data does not match Database. "
            "Combine synonyms on one line: Sociotechnical = socio-technical, socio technical. "
            "The line adds every form. Overlapping text counts once, keeping the longer form."
        ),
    )

    if not run:
        st.info(f"Loaded {len(corpus):,} papers, {year_lo}–{year_hi}. Choose filters, then calculate.")
        return

    try:
        terms = ais_ngram.parse_term_groups(terms_text)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not terms:
        st.error("Enter at least one word or phrase.")
        return
    if not types:
        st.error("Select at least one document type.")
        return
    if not fields:
        st.error("Select at least one field.")
        return

    with st.spinner("Calculating..."):
        result = run_query(
            terms,
            year_min=year_min,
            year_max=year_max,
            types=types,
            venue=venue,
            fields=fields,
            corpus=corpus,
        )
        charts = ais_ngram.plot_percentages(
            result,
            terms,
            denominator=denominator,
            title=_chart_title(denominator, fields),
            caption=_caption(result, denominator, fields),
        )

    value_column = {"tokens": "pct_tokens", "all": "pct_all", "with_text": "pct_with_text"}[denominator]
    digits = 4 if denominator == "tokens" else 2
    labels = [group.display for group in terms]
    wide = result.pivot(index="year", columns="term", values=value_column)
    wide = wide.reindex(columns=labels)
    base = result.groupby("year", sort=True)[["n_papers", "n_with_text", "n_tokens"]].first()
    table = base.join(wide)
    table.index.name = "year"
    table[labels] = table[labels].round(digits)

    st.download_button(
        "Download CSV",
        data=result.to_csv(index=False, float_format="%.6f").encode("utf-8-sig"),
        file_name="ais_ngram.csv",
        mime="text/csv",
    )
    for label, _path, png in charts:
        st.subheader(label)
        st.image(png, width="stretch")
    st.dataframe(table, width="stretch")


if __name__ == "__main__":
    main()
