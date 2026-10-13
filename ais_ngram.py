"""AIS Ngram: yearly word share in titles, abstracts, and keywords."""

from __future__ import annotations

import os
import pickle
import re
import textwrap
import unicodedata
from pathlib import Path
from typing import NamedTuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "aisel_all_papers_20260928.xlsx"
CACHE_PATH = ROOT / "cache" / "corpus.pkl"
CACHE_VERSION = 3
_WORD = re.compile(r"[^\W_]+")

# Separates fields so a phrase cannot start in one field and finish in another.
_FIELD_BREAK = "\u0001"
_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_WS = re.compile(r"\s+")
# Letters or digits in any script. Underscore is a boundary.
_EDGE = r"[^\W_]"
_TEXT_FIELDS = ("title", "abstract", "keywords")


def prepare_text(value) -> str:
    """Lowercase, collapse whitespace, and normalize dashes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace(_FIELD_BREAK, " ")
    text = _DASHES.sub("-", text)
    text = text.lower()
    text = _WS.sub(" ", text).strip()
    return text


def count_words(text: str) -> int:
    """Split on punctuation and hyphens, so socio-technical is two words."""
    if not text:
        return 0
    return len(_WORD.findall(text))


class TermGroup(NamedTuple):
    """One plotted line. forms are the spellings whose counts are added."""

    label: str
    forms: tuple[str, ...]

    @property
    def display(self) -> str:
        extras = [form for form in self.forms if prepare_text(form) != prepare_text(self.label)]
        if not extras:
            return self.label
        return f"{self.label} (+ {', '.join(extras)})"


def _form_body(term: str) -> str:
    prepared = prepare_text(term)
    tokens = prepared.split(" ")
    if not tokens or tokens == [""]:
        raise ValueError("A term cannot be empty")
    return r"\s+".join(re.escape(token) for token in tokens)


def term_pattern(term: str) -> re.Pattern[str]:
    """Whole word or whole phrase. Data does not match Database or plurals."""
    return re.compile(rf"(?<!{_EDGE}){_form_body(term)}(?!{_EDGE})")


def group_pattern(forms: tuple[str, ...] | list[str]) -> re.Pattern[str]:
    """Match any form. A longer form wins when two forms overlap, so it is counted once."""
    ordered = sorted(forms, key=lambda form: len(prepare_text(form)), reverse=True)
    alt = "|".join(_form_body(form) for form in ordered)
    return re.compile(rf"(?<!{_EDGE})(?:{alt})(?!{_EDGE})")


def parse_term_groups(text: str) -> list[TermGroup]:
    """One term per line. Blank lines are ignored.

    Synonyms: ``Sociotechnical = socio-technical, socio technical``.
    The name before = is included, and each form after = is added to the same line.
    """
    seen: set[str] = set()
    groups: list[TermGroup] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if "=" in raw:
            label, rest = raw.split("=", 1)
            label = label.strip()
            extras = [part.strip() for part in rest.split(",") if part.strip()]
            if not label:
                raise ValueError("Write the term before =")
        else:
            label = raw
            extras = []
        forms: list[str] = []
        form_keys: set[str] = set()
        for form in (label, *extras):
            key = prepare_text(form)
            if not key or key in form_keys:
                continue
            form_keys.add(key)
            forms.append(form)
        label_key = prepare_text(label)
        if not forms or not label_key or label_key in seen:
            continue
        seen.add(label_key)
        groups.append(TermGroup(label, tuple(forms)))
    return groups


def parse_terms(text: str) -> list[str]:
    """Labels only, in order. Duplicate labels keep the first line."""
    return [group.label for group in parse_term_groups(text)]


def _as_groups(terms: list[TermGroup] | list[str]) -> list[TermGroup]:
    groups: list[TermGroup] = []
    seen: set[str] = set()
    for term in terms:
        parsed = [term] if isinstance(term, TermGroup) else parse_term_groups(str(term))
        for group in parsed:
            key = prepare_text(group.label)
            if not key or key in seen:
                continue
            seen.add(key)
            groups.append(group)
    return groups


def load_corpus(path: Path = DATA_PATH, cache_path: Path = CACHE_PATH, refresh: bool = False) -> pd.DataFrame:
    """Load the Papers sheet and cache normalized title, abstract, and keywords."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    stat = path.stat()
    if cache_path.exists() and not refresh:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        frame = payload.get("df")
        same_file = payload.get("mtime") == stat.st_mtime and payload.get("size") == stat.st_size
        if same_file and frame is not None and set(_TEXT_FIELDS).issubset(frame.columns):
            token_columns = {f"n_{field}" for field in _TEXT_FIELDS}
            if not token_columns.issubset(frame.columns):
                frame = _attach_token_counts(frame)
                payload = {
                    "version": CACHE_VERSION,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "df": frame,
                }
                with cache_path.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return frame

    raw = pd.read_excel(
        path,
        sheet_name="Papers",
        usecols=["Title", "Year", "Type", "Category", "Venue", "Abstract", "Keywords"],
    )
    title = raw["Title"].map(prepare_text)
    abstract = raw["Abstract"].map(prepare_text)
    keywords = raw["Keywords"].map(prepare_text)
    frame = pd.DataFrame(
        {
            "year": raw["Year"].astype(int),
            "type": raw["Type"].fillna("").astype(str),
            "category": raw["Category"].fillna("").astype(str),
            "venue": raw["Venue"].fillna("").astype(str),
            "title": title,
            "abstract": abstract,
            "keywords": keywords,
            "has_text": (title.str.len() > 0) | (abstract.str.len() > 0) | (keywords.str.len() > 0),
        }
    )
    frame = _attach_token_counts(frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "df": frame,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return frame


def _attach_token_counts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["n_tokens"] = 0
    for field in _TEXT_FIELDS:
        counts = frame[field].map(count_words)
        frame[f"n_{field}"] = counts
        frame["n_tokens"] = frame["n_tokens"] + counts
    return frame


def normalize_fields(fields: list[str] | None) -> list[str]:
    """None means all fields. An empty list is an error."""
    if fields is None:
        return list(_TEXT_FIELDS)
    unknown = [field for field in fields if field not in _TEXT_FIELDS]
    if unknown:
        raise ValueError("Unknown field: " + ", ".join(unknown))
    selected = [field for field in _TEXT_FIELDS if field in fields]
    if not selected:
        raise ValueError("Select at least one field")
    return selected


def filter_corpus(
    corpus: pd.DataFrame,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    types: list[str] | None = None,
    venue: str | None = None,
) -> pd.DataFrame:
    frame = corpus
    if year_min is not None:
        frame = frame[frame["year"] >= year_min]
    if year_max is not None:
        frame = frame[frame["year"] <= year_max]
    if types:
        frame = frame[frame["type"].isin(types)]
    venue_text = (venue or "").strip()
    if venue_text:
        frame = frame[frame["venue"].str.contains(venue_text, case=False, regex=False, na=False)]
    return frame


def query_terms(
    frame: pd.DataFrame,
    terms: list[TermGroup] | list[str],
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Long table with word share and paper share for each term and year.

    pct_tokens: occurrences / all words in the selected fields that year.
    Repeated occurrences in one paper are counted again.
    A synonym line adds the occurrences of each form. Overlapping forms count once,
    and the longer form is kept.
    pct_all: papers with at least one occurrence / all papers that year.
    pct_with_text: papers with at least one occurrence / papers that have text in a selected field.
    A paper counts once when any synonym form appears.
    """
    if frame.empty:
        raise ValueError("No papers match the current filters")
    groups = _as_groups(terms)
    if not groups:
        raise ValueError("Enter at least one word or phrase")
    selected = normalize_fields(fields)
    if any(field not in frame.columns or f"n_{field}" not in frame.columns for field in selected):
        frame = _attach_token_counts(frame)

    token_counts = frame[f"n_{selected[0]}"]
    has_text = frame[selected[0]].str.len().gt(0)
    for field in selected[1:]:
        token_counts = token_counts + frame[f"n_{field}"]
        has_text = has_text | frame[field].str.len().gt(0)
    counts = pd.DataFrame(
        {"year": frame["year"], "has_text": has_text, "n_tokens": token_counts}
    ).groupby("year", sort=True).agg(
        n_papers=("year", "size"),
        n_with_text=("has_text", "sum"),
        n_tokens=("n_tokens", "sum"),
    )
    pieces: list[pd.DataFrame] = []
    labels = [group.display for group in groups]
    for group in groups:
        pattern = group_pattern(group.forms)
        occurrences = sum(frame[field].str.count(pattern) for field in selected)
        n_occurrences = occurrences.groupby(frame["year"], sort=True).sum().astype(int)
        n_hit = (occurrences > 0).groupby(frame["year"], sort=True).sum().astype(int)
        part = counts.copy()
        part["n_occurrences"] = n_occurrences.reindex(part.index).fillna(0).astype(int)
        part["n_hit"] = n_hit.reindex(part.index).fillna(0).astype(int)
        part["term"] = group.display
        part["forms"] = " | ".join(group.forms)
        pieces.append(part.reset_index())

    result = pd.concat(pieces, ignore_index=True)
    result["pct_tokens"] = float("nan")
    has_tokens = result["n_tokens"] > 0
    result.loc[has_tokens, "pct_tokens"] = (
        result.loc[has_tokens, "n_occurrences"] / result.loc[has_tokens, "n_tokens"] * 100
    )
    result["pct_all"] = result["n_hit"] / result["n_papers"] * 100
    result["pct_with_text"] = float("nan")
    has_base = result["n_with_text"] > 0
    result.loc[has_base, "pct_with_text"] = (
        result.loc[has_base, "n_hit"] / result.loc[has_base, "n_with_text"] * 100
    )
    result["term"] = pd.Categorical(result["term"], categories=labels, ordered=True)
    result = result.sort_values(["term", "year"]).reset_index(drop=True)
    return result[
        [
            "term",
            "forms",
            "year",
            "n_papers",
            "n_with_text",
            "n_tokens",
            "n_hit",
            "n_occurrences",
            "pct_tokens",
            "pct_all",
            "pct_with_text",
        ]
    ]


VALUE_COLUMNS = {
    "tokens": ("pct_tokens", "Percentage of all words (%)"),
    "all": ("pct_all", "Percentage of all papers (%)"),
    "with_text": ("pct_with_text", "Percentage of papers with text (%)"),
}

# Fixed colors for the comparison set. Other terms take the fallback palette.
_KNOWN_STYLE = {
    "algorithm": ("#0072B2", "-"),
    "bit": ("#E69F00", "-"),
    "data": ("#009E73", "-"),
    "database": ("#009E73", "--"),
    "decision": ("#D55E00", "-"),
    "hardware": ("#2E86AB", "-"),
    "information": ("#111111", "-"),
    "information systems": ("#111111", "--"),
    "information technology": ("#111111", ":"),
    "knowledge": ("#CC79A7", "-"),
    "sociotechnical": ("#882255", "-"),
    "software": ("#7B4B94", "-"),
}
_FALLBACK_COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#EE7733",
    "#009988",
    "#CC3311",
    "#332288",
    "#117733",
    "#882255",
]
_FALLBACK_LINES = ["-", "--", "-.", ":"]


def _style_for(term: str, index: int) -> tuple[str, str]:
    known = _KNOWN_STYLE.get(prepare_text(term.split(" (+ ", 1)[0]))
    if known:
        return known
    return (
        _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)],
        _FALLBACK_LINES[(index // len(_FALLBACK_COLORS)) % len(_FALLBACK_LINES)],
    )


def _setup_matplotlib():
    mpl_dir = ROOT / "cache" / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
        }
    )
    return plt


def _term_groups(wide: pd.DataFrame, terms: list[str]) -> list[tuple[str, list[str]]]:
    """Split terms into two charts when their percentages differ a lot.

    A term goes to the lower chart when its 90th-percentile yearly percentage
    is at most 20% of the highest term. Otherwise it stays on the higher chart.
    """
    typical = wide.quantile(0.9)
    finite = typical.dropna()
    highest = float(finite.max()) if not finite.empty else 0.0
    if highest <= 0:
        return [("Word share", terms)]
    low_terms = [
        term
        for term in terms
        if pd.notna(typical.get(term)) and float(typical[term]) <= highest * 0.2
    ]
    high_terms = [term for term in terms if term not in set(low_terms)]
    if not high_terms or not low_terms:
        return [("Word share", terms)]
    return [("Higher share", high_terms), ("Lower share", low_terms)]


def _chart_path(path: Path, label: str, groups: int) -> Path:
    if groups == 1:
        return path
    suffix = "_higher" if label == "Higher share" else "_lower"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def plot_percentages(
    result: pd.DataFrame,
    terms: list[TermGroup] | list[str],
    *,
    denominator: str = "tokens",
    title: str | None = None,
    caption: str | None = None,
    path: Path | None = None,
) -> list[tuple[str, Path | None, bytes]]:
    """Draw one line chart per magnitude group. Returns (label, path, png bytes)."""
    if denominator not in VALUE_COLUMNS:
        raise ValueError("denominator must be tokens, all, or with_text")
    terms = [group.display for group in _as_groups(terms)]
    value_col, ylabel = VALUE_COLUMNS[denominator]
    plt = _setup_matplotlib()
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    import io

    wide = result.pivot(index="year", columns="term", values=value_col)
    wide = wide.reindex(columns=terms)
    years = wide.index.to_numpy()
    groups = _term_groups(wide, terms)
    default_title = (
        "Word share in AISeL titles, abstracts, and keywords"
        if denominator == "tokens"
        else "Share of AISeL papers containing the term"
    )
    outputs: list[tuple[str, Path | None, bytes]] = []

    for label, panel_terms in groups:
        fig, ax = plt.subplots(figsize=(12.2, 6.4))
        for term in panel_terms:
            color, linestyle = _style_for(term, terms.index(term))
            ax.plot(
                years,
                wide[term].to_numpy(dtype=float),
                label=term,
                color=color,
                linestyle=linestyle,
                linewidth=1.9,
            )
        values = wide[panel_terms].to_numpy(dtype=float)
        values = values[pd.notna(values)]
        top = float(values.max()) if len(values) else 1.0
        ax.set_ylim(0, top * 1.12 if top > 0 else 1)
        if top < 0.05:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        elif top < 1:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        elif top < 10:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if len(years):
            span = int(years.max() - years.min())
            step = 5 if span > 25 else 2 if span > 12 else 1
            ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.set_xlim(int(years.min()) - 0.5, int(years.max()) + 0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        heading = title or default_title
        if len(groups) > 1:
            heading = f"{heading}: {label.lower()}"
        ax.set_title(heading)
        ax.yaxis.grid(True, color="#d0d0d0", linewidth=0.6)
        ax.xaxis.grid(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), borderaxespad=0)
        note = caption or ""
        if len(groups) > 1:
            scope = (
                "above 20% of the highest term"
                if label == "Higher share"
                else "at or below 20% of the highest term"
            )
            note = (
                f"This chart includes terms whose 90th-percentile yearly percentage is {scope}. "
                + note
            )
        wrapped = ""
        bottom = 0.02
        if note:
            wrapped = "\n".join(textwrap.wrap(note, width=108))
            bottom = 0.012 + 0.028 * (wrapped.count("\n") + 1)
        fig.tight_layout(rect=(0, bottom, 0.84, 1))
        if wrapped:
            fig.text(0.01, 0.006, wrapped, ha="left", va="bottom", fontsize=8.5, color="#333333")

        written: Path | None = None
        if path is not None:
            written = _chart_path(path, label, len(groups))
            written.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(written, dpi=180, bbox_inches="tight", facecolor="white")
            if written.suffix.lower() == ".png":
                fig.savefig(written.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs.append((label, written, buffer.getvalue()))
    return outputs


def write_result_csv(result: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def run_self_check() -> None:
    cases = [
        ("We study the database and data quality.", "keywords", "data", True),
        ("We study the database and data quality.", "keywords", "database", True),
        ("We study the database and data quality.", "keywords", "databases", False),
        ("information systems research", "", "information systems", True),
        ("information systems research", "", "information technology", False),
        ("information systems research", "", "information", True),
        ("This is a habit of 8-bit design.", "", "bit", True),
        ("This is a habit.", "", "bit", False),
        ("This is a habit of 8-bit design.", "", "habit", True),
        ("decisions about software", "", "decision", False),
        ("decisions about software", "", "software", True),
        ("INFORMATION SYSTEMS", "", "Information systems", True),
        ("information", "systems", "information systems", False),
        ("socio-technical design", "", "sociotechnical", False),
        ("sociotechnical design", "", "sociotechnical", True),
    ]
    failures: list[str] = []
    if count_words("information systems research") != 3:
        failures.append("word count failed")
    if len(term_pattern("information").findall("information and information systems")) != 2:
        failures.append("occurrence count failed")
    if not term_pattern("information systems").search(prepare_text("Advances in Information Systems")):
        failures.append("title text should match the phrase")
    for abstract, keywords, term, expected in cases:
        pattern = term_pattern(term)
        hit = bool(pattern.search(prepare_text(abstract)) or pattern.search(prepare_text(keywords)))
        if hit != expected:
            failures.append(f"{term!r} in {abstract!r} / {keywords!r}: got {hit}, expected {expected}")
    combined = group_pattern(("information", "information systems"))
    combined_text = prepare_text("information systems and information")
    if len(combined.findall(combined_text)) != 2:
        failures.append("overlapping synonyms should count the longer form once")
    groups = parse_term_groups("Sociotechnical = socio-technical, socio-technical\nData\n\nData")
    if [group.label for group in groups] != ["Sociotechnical", "Data"]:
        failures.append("synonym lines were not parsed")
    if groups[0].forms != ("Sociotechnical", "socio-technical"):
        failures.append(f"unexpected forms: {groups[0].forms}")
    sample = pd.DataFrame(
        {
            "year": [2020, 2020],
            "title": [prepare_text("Sociotechnical design"), prepare_text("A socio-technical study")],
            "abstract": ["", ""],
            "keywords": ["", ""],
        }
    )
    sample = _attach_token_counts(sample)
    combined_result = query_terms(sample, groups[:1], ["title"])
    if int(combined_result["n_occurrences"].iloc[0]) != 2 or int(combined_result["n_hit"].iloc[0]) != 2:
        failures.append("synonym counts were not added")
    nested = pd.DataFrame(
        {
            "year": [2020],
            "title": [prepare_text("Information systems")],
            "abstract": [""],
            "keywords": [prepare_text("information")],
        }
    )
    nested = _attach_token_counts(nested)
    nested_result = query_terms(nested, parse_term_groups("Information = information systems"), ["title", "keywords"])
    if int(nested_result["n_occurrences"].iloc[0]) != 2 or int(nested_result["n_hit"].iloc[0]) != 1:
        failures.append("nested synonyms were double-counted")
    if failures:
        raise AssertionError("Match check failed:\n" + "\n".join(failures))
