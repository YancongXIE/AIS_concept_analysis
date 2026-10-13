#!/usr/bin/env python3
"""Local AIS Ngram app.

Yearly share of a word or phrase in titles, abstracts, and keywords.

Usage:
    python app.py serve
    python app.py query Algorithm Bit "Information systems" --out results/selected_terms
    python app.py check
"""

from __future__ import annotations

import argparse
import base64
import html
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

import ais_ngram

ROOT = Path(__file__).resolve().parent
_COMPUTE_LOCK = threading.Lock()
DEFAULT_TERMS = """Algorithm
Bit
Data
Database
Decision
Hardware
Information
Information systems
Information technology
Knowledge
Sociotechnical
Software
"""


def _field_phrase(fields: list[str] | None) -> str:
    labels = {"title": "titles", "abstract": "abstracts", "keywords": "keywords"}
    names = [labels[field] for field in ais_ngram.normalize_fields(fields)]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {names[2]}"


def _chart_title(denominator: str, fields: list[str] | None) -> str:
    phrase = _field_phrase(fields)
    if denominator == "tokens":
        return f"Word share in AISeL {phrase}"
    return f"Share of AISeL papers containing the term in {phrase}"


def _caption(result: pd.DataFrame, denominator: str, fields: list[str] | None) -> str:
    year_min = int(result["year"].min())
    year_max = int(result["year"].max())
    one = result[result["term"] == result["term"].iloc[0]]
    n_papers = int(one["n_papers"].sum())
    n_tokens = int(one["n_tokens"].sum()) if "n_tokens" in one.columns else 0
    phrase = _field_phrase(fields)
    if denominator == "tokens":
        denom_note = (
            f"Percentage = occurrences divided by all words in {phrase} that year. "
            "Repeated occurrences in one paper count again. This is the Google Ngram word share. "
        )
    elif denominator == "all":
        denom_note = (
            "Percentage = papers with at least one occurrence divided by all papers that year. "
            f"A paper with no text in {phrase} counts as a non-match. "
        )
    else:
        denom_note = (
            f"Percentage = papers with at least one occurrence divided by papers that have text in {phrase}. "
            "Years with no text are omitted. "
        )
    return (
        f"{denom_note}"
        "Matching is case-insensitive and whole-word: Data does not include Database or plurals. "
        "Information includes the word inside Information systems. "
        "A phrase must sit entirely inside one field. "
        f"Fields: {', '.join(ais_ngram.normalize_fields(fields))}. "
        f"Sample: {year_min}–{year_max}, {n_papers:,} papers and {n_tokens:,} words. "
        "Source: aisel_all_papers_20260928, exported 2026-09-28; 2026 is a partial year."
    )


def run_query(
    terms: list[str],
    *,
    year_min: int | None,
    year_max: int | None,
    types: list[str] | None,
    venue: str | None,
    fields: list[str] | None = None,
    corpus: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = corpus if corpus is not None else ais_ngram.load_corpus()
    filtered = ais_ngram.filter_corpus(
        frame, year_min=year_min, year_max=year_max, types=types, venue=venue
    )
    return ais_ngram.query_terms(filtered, terms, fields)


def save_query(
    result: pd.DataFrame,
    terms: list[str],
    out: Path,
    denominator: str,
    fields: list[str] | None = None,
) -> tuple[Path, list[Path]]:
    csv_path = out if out.suffix.lower() == ".csv" else out.with_suffix(".csv")
    ais_ngram.write_result_csv(result, csv_path)
    charts = ais_ngram.plot_percentages(
        result,
        terms,
        denominator=denominator,
        title=_chart_title(denominator, fields),
        caption=_caption(result, denominator, fields),
        path=csv_path.with_suffix(".png"),
    )
    return csv_path, [path for _, path, _ in charts if path is not None]


def _page(body: str) -> bytes:
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIS Ngram</title>
<style>
  body {{
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    margin: 28px auto 64px;
    max-width: 1100px;
    padding: 0 20px;
    color: #1c1c1c;
    line-height: 1.5;
    background: #fff;
  }}
  h1 {{ font-size: 28px; font-weight: 650; margin: 0 0 6px; }}
  .sub {{ margin: 0 0 18px; color: #444; }}
  label {{ display: block; font-weight: 650; margin: 16px 0 6px; }}
  .hint {{ font-weight: 400; color: #555; font-size: 13px; }}
  textarea {{
    width: 100%; box-sizing: border-box; min-height: 220px;
    font: 15px/1.45 ui-monospace, "SF Mono", Menlo, monospace;
    padding: 10px; border: 1px solid #ccc;
  }}
  input[type="number"], input[type="text"] {{
    font: 15px sans-serif; padding: 6px 8px; border: 1px solid #ccc; width: 140px;
  }}
  .row {{ display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-end; }}
  .checks label, .radios label {{ font-weight: 450; display: inline-flex; gap: 6px; margin: 0 14px 0 0; }}
  button {{
    margin-top: 18px; background: #1c1c1c; color: #fff; border: 0;
    padding: 10px 18px; font-size: 15px; cursor: pointer;
  }}
  button:disabled {{ background: #666; }}
  .error {{ color: #8a1f1f; white-space: pre-wrap; }}
  img {{ width: 100%; margin-top: 8px; }}
  .table-wrap {{ overflow: auto; max-height: 520px; border: 1px solid #e6e6e6; margin-top: 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 5px 8px; text-align: right; white-space: nowrap; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  thead th {{ position: sticky; top: 0; background: #fafafa; }}
  a {{ color: #1c1c1c; }}
</style>
</head>
<body>
{body}
<script>
const form = document.querySelector("form");
if (form) form.addEventListener("submit", () => {{
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Calculating…";
}});
</script>
</body>
</html>
"""
    return document.encode("utf-8")


def _form_html(
    *,
    year_min_bound: int,
    year_max_bound: int,
    terms: str,
    year_min: str,
    year_max: str,
    types: list[str],
    fields: list[str],
    venue: str,
    denominator: str,
    error: str = "",
    images: list[tuple[str, bytes]] | None = None,
    result: pd.DataFrame | None = None,
) -> str:
    def checked_type(name: str) -> str:
        return " checked" if name in types else ""

    def checked_field(name: str) -> str:
        return " checked" if name in fields else ""

    def checked_denom(name: str) -> str:
        return " checked" if denominator == name else ""

    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    result_html = ""
    if images and result is not None:
        csv_text = result.to_csv(index=False, float_format="%.6f")
        csv_href = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_text)
        figures = []
        for label, image in images:
            encoded = base64.b64encode(image).decode("ascii")
            figures.append(
                f'<h2>{html.escape(label)}</h2>'
                f'<img alt="{html.escape(label)} line chart" src="data:image/png;base64,{encoded}">'
            )
        result_html = (
            f'<h2>Results</h2><p><a download="ais_ngram.csv" href="{csv_href}">Download CSV</a></p>'
            + "".join(figures)
            + f'<div class="table-wrap">{_table_html(result, denominator)}</div>'
        )
    return f"""
<h1>AIS Ngram</h1>
<p class="sub">Share of a word or phrase among all words in titles, abstracts, and keywords in <code>aisel_all_papers_20260928</code>. The default matches Google Ngram: occurrences divided by total words. One mention in a paper counts only those occurrences.</p>
<form method="post" action="/query">
  <label>Words or phrases <span class="hint">One per line. Case-insensitive whole words: Data does not match Database.</span></label>
  <textarea name="terms">{html.escape(terms)}</textarea>
  <div class="row">
    <div>
      <label>Start year</label>
      <input type="number" name="year_min" value="{html.escape(year_min)}" min="{year_min_bound}" max="{year_max_bound}">
    </div>
    <div>
      <label>End year</label>
      <input type="number" name="year_max" value="{html.escape(year_max)}" min="{year_min_bound}" max="{year_max_bound}">
    </div>
    <div>
      <label>Venue contains <span class="hint">optional</span></label>
      <input type="text" name="venue" value="{html.escape(venue)}" style="width:280px">
    </div>
  </div>
  <label>Document type</label>
  <div class="checks">
    <label><input type="checkbox" name="types" value="Conference"{checked_type("Conference")}>Conference</label>
    <label><input type="checkbox" name="types" value="Journal"{checked_type("Journal")}>Journal</label>
    <label><input type="checkbox" name="types" value="Other"{checked_type("Other")}>Other</label>
  </div>
  <label>Fields</label>
  <div class="checks">
    <label><input type="checkbox" name="fields" value="title"{checked_field("title")}>Title</label>
    <label><input type="checkbox" name="fields" value="abstract"{checked_field("abstract")}>Abstract</label>
    <label><input type="checkbox" name="fields" value="keywords"{checked_field("keywords")}>Keywords</label>
  </div>
  <label>Percentage</label>
  <div class="radios">
    <label><input type="radio" name="denominator" value="tokens"{checked_denom("tokens")}>Word share: occurrences / all words</label>
    <label><input type="radio" name="denominator" value="with_text"{checked_denom("with_text")}>Paper share: papers with text that contain the term</label>
    <label><input type="radio" name="denominator" value="all"{checked_denom("all")}>Paper share: all papers that contain the term</label>
  </div>
  <button type="submit">Calculate and plot</button>
</form>
{error_html}
{result_html}
"""


def _table_html(result: pd.DataFrame, denominator: str) -> str:
    value = {"tokens": "pct_tokens", "all": "pct_all", "with_text": "pct_with_text"}[denominator]
    digits = 4 if denominator == "tokens" else 2
    wide = result.pivot(index="year", columns="term", values=value)
    base_cols = ["n_papers", "n_with_text"] + (["n_tokens"] if "n_tokens" in result.columns else [])
    base = result.groupby("year", sort=True)[base_cols].first()
    header = "".join(f"<th>{html.escape(str(term))}</th>" for term in wide.columns)
    token_head = "<th>Words</th>" if "n_tokens" in base.columns else ""
    rows = []
    for year, values in wide.iterrows():
        cells = "".join(
            f"<td>{'' if pd.isna(item) else f'{float(item):.{digits}f}'}</td>" for item in values.tolist()
        )
        token_cell = f"<td>{int(base.loc[year, 'n_tokens'])}</td>" if "n_tokens" in base.columns else ""
        rows.append(
            "<tr>"
            f"<td>{int(year)}</td>"
            f"<td>{int(base.loc[year, 'n_papers'])}</td>"
            f"<td>{int(base.loc[year, 'n_with_text'])}</td>"
            f"{token_cell}{cells}</tr>"
        )
    return (
        "<table><thead><tr><th>Year</th><th>Papers</th><th>With text</th>"
        f"{token_head}{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


class App:
    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus


def make_handler(app: App):
    year_min_bound = int(app.corpus["year"].min())
    year_max_bound = int(app.corpus["year"].max())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print("[ais-ngram]", fmt % args)

        def _send(self, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if urllib.parse.urlparse(self.path).path not in ("/", "/index.html"):
                self.send_error(404)
                return
            page = _form_html(
                year_min_bound=year_min_bound,
                year_max_bound=year_max_bound,
                terms=DEFAULT_TERMS,
                year_min=str(year_min_bound),
                year_max=str(year_max_bound),
                types=["Conference", "Journal", "Other"],
                fields=["title", "abstract", "keywords"],
                venue="",
                denominator="tokens",
            )
            self._send(_page(page))

        def do_POST(self) -> None:  # noqa: N802
            if urllib.parse.urlparse(self.path).path != "/query":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            fields = urllib.parse.parse_qs(raw, keep_blank_values=True)

            def first(name: str, default: str = "") -> str:
                return fields.get(name, [default])[0]

            terms_text = first("terms", DEFAULT_TERMS)
            year_min_text = first("year_min", str(year_min_bound))
            year_max_text = first("year_max", str(year_max_bound))
            venue = first("venue")
            denominator = first("denominator", "tokens")
            types = fields.get("types", [])
            text_fields = fields.get("fields", [])
            error = ""
            images = None
            result = None
            try:
                if denominator not in ("tokens", "all", "with_text"):
                    raise ValueError("Choose word share or paper share")
                if not types:
                    raise ValueError("Select at least one document type")
                if not text_fields:
                    raise ValueError("Select at least one field")
                terms = ais_ngram.parse_terms(terms_text)
                if not terms:
                    raise ValueError("Enter at least one word or phrase")
                year_min = int(year_min_text)
                year_max = int(year_max_text)
                if year_min > year_max:
                    raise ValueError("Start year cannot be later than end year")
                with _COMPUTE_LOCK:
                    result = run_query(
                        terms,
                        year_min=year_min,
                        year_max=year_max,
                        types=types,
                        venue=venue,
                        fields=text_fields,
                        corpus=app.corpus,
                    )
                    charts = ais_ngram.plot_percentages(
                        result,
                        terms,
                        denominator=denominator,
                        title=_chart_title(denominator, text_fields),
                        caption=_caption(result, denominator, text_fields),
                    )
                    images = [(label, png) for label, _, png in charts]
            except Exception as exc:  # noqa: BLE001 — show the calculation error on the page
                error = str(exc).strip() or traceback.format_exc()
            page = _form_html(
                year_min_bound=year_min_bound,
                year_max_bound=year_max_bound,
                terms=terms_text,
                year_min=year_min_text,
                year_max=year_max_text,
                types=types,
                fields=text_fields,
                venue=venue,
                denominator=denominator if denominator in ("tokens", "all", "with_text") else "tokens",
                error=error,
                images=images,
                result=result,
            )
            self._send(_page(page))

    return Handler


def serve(port: int) -> None:
    print("Loading corpus...", flush=True)
    corpus = ais_ngram.load_corpus()
    print(
        f"Loaded {len(corpus):,} papers, years {int(corpus['year'].min())}–{int(corpus['year'].max())}",
        flush=True,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(App(corpus)))
    print(f"AIS Ngram is running at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AIS Ngram: yearly word share in titles, abstracts, and keywords")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Run the matching checks")
    check.set_defaults(command="check")

    query = sub.add_parser("query", help="Calculate terms and save a CSV plus line charts")
    query.add_argument("terms", nargs="*", help="Words or phrases. Quote a phrase that contains spaces")
    query.add_argument("--terms-file", type=Path, help="Text file with one term per line")
    query.add_argument("--year-min", type=int)
    query.add_argument("--year-max", type=int)
    query.add_argument("--types", nargs="*", choices=["Conference", "Journal", "Other"])
    query.add_argument(
        "--fields",
        nargs="*",
        choices=["title", "abstract", "keywords"],
        help="Fields to search. Defaults to title, abstract, and keywords",
    )
    query.add_argument("--venue", default="", help="Substring of the venue name")
    query.add_argument("--denominator", choices=["tokens", "all", "with_text"], default="tokens")
    query.add_argument("--out", type=Path, default=ROOT / "results" / "query")

    server = sub.add_parser("serve", help="Start the local web page")
    server.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    if args.command == "check":
        ais_ngram.run_self_check()
        print("Match check passed")
        return
    if args.command == "serve":
        serve(args.port)
        return

    chunks = list(args.terms)
    if args.terms_file:
        chunks.extend(ais_ngram.parse_terms(args.terms_file.read_text(encoding="utf-8")))
    terms = ais_ngram.parse_terms("\n".join(chunks))
    if not terms:
        parser.error("Provide at least one term as an argument or with --terms-file")
    print("Loading corpus...")
    result = run_query(
        terms,
        year_min=args.year_min,
        year_max=args.year_max,
        types=args.types,
        venue=args.venue,
        fields=args.fields,
    )
    csv_path, chart_paths = save_query(result, terms, args.out, args.denominator, args.fields)
    print(f"CSV: {csv_path}")
    for chart_path in chart_paths:
        print(f"Chart: {chart_path}")
        print(f"PDF: {chart_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
