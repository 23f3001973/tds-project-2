#!/usr/bin/env python3
"""
Data Analyst Agent - universal API endpoint
POST /api/ with multipart form:
 - questions.txt (required)
 - optional files: data.csv, data.json, file.parquet, images...
Returns JSON in the format requested by the question text (array or object).
"""
import os
import re
import time
import json
import base64
import traceback
from io import BytesIO

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from bs4 import BeautifulSoup

# Optional: duckdb for parquet & SQL
try:
    import duckdb
    DUCKDB_OK = True
except Exception:
    DUCKDB_OK = False

# Optional: OpenAI for task parsing (if OPENAI_API_KEY set)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
try:
    if OPENAI_API_KEY:
        import openai
        openai.api_key = OPENAI_API_KEY
        LLM_OK = True
    else:
        LLM_OK = False
except Exception:
    LLM_OK = False

app = Flask(__name__)

# ---------- CONFIG ----------
MAX_ROWS_FOR_PLOT = 2000      # sample large tables for plotting
PLOT_MAX_BYTES = 100_000      # required by evaluation
REQUEST_TIMEOUT = 40          # seconds for external requests
MAX_DATA_ROWS = 200_000       # safety cap for in-memory operations
# ----------------------------

# ---------- Utilities ----------
def now_ts():
    return int(time.time())

def extract_urls(text):
    # match http(s) and s3
    return re.findall(r'(https?://[^\s,]+|s3://[^\s,]+)', text)

def safe_read_csv_bytes(b):
    try:
        return pd.read_csv(BytesIO(b))
    except Exception:
        # try with utf-8-sig or latin1
        for enc in ("utf-8-sig", "latin1"):
            try:
                return pd.read_csv(BytesIO(b), encoding=enc)
            except Exception:
                pass
    raise

def try_read_json_bytes(b):
    try:
        return pd.read_json(BytesIO(b))
    except Exception:
        # try json lines
        try:
            return pd.read_json(BytesIO(b), lines=True)
        except Exception:
            raise

def fetch_url_table(url):
    """
    Try to fetch a table from a URL: CSV, JSON, HTML tables, or duckdb read_parquet for s3.
    Returns DataFrame or raw text on fallback.
    """
    # CSV
    if re.search(r'\.csv($|\?)', url, re.I):
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return safe_read_csv_bytes(r.content)

    # JSON
    if re.search(r'\.json($|\?)', url, re.I):
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return try_read_json_bytes(r.content)

    # parquet / s3 (duckdb)
    if (url.endswith(".parquet") or url.startswith("s3://")) and DUCKDB_OK:
        try:
            # read into duckdb then to pandas
            q = f"SELECT * FROM read_parquet('{url}')"
            return duckdb.query(q).to_df()
        except Exception as e:
            # fallback to trying http fetch
            pass

    # try pandas read_html
    try:
        tables = pd.read_html(url, attrs={"role": None}, flavor='bs4')
        if tables:
            return tables[0] if len(tables) == 1 else tables
    except Exception:
        pass

    # final fallback: raw text
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def load_attachments(files):
    """
    Accepts Flask files dict-like. Returns list of DataFrames or text blobs.
    """
    datasets = []
    for key in files:
        # skip questions.txt
        if key.lower() == "questions.txt":
            continue
        file = files.get(key)
        fname = getattr(file, "filename", key)
        content = file.read()
        # small heuristics by extension
        if fname.lower().endswith(".csv"):
            try:
                df = safe_read_csv_bytes(content)
                datasets.append(("file", fname, df))
                continue
            except Exception:
                pass
        if fname.lower().endswith(".json") or fname.lower().endswith(".ndjson") or fname.lower().endswith(".jsonl"):
            try:
                df = try_read_json_bytes(content)
                datasets.append(("file", fname, df))
                continue
            except Exception:
                pass
        if fname.lower().endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                bio = BytesIO(content)
                df = pd.read_parquet(bio)
                datasets.append(("file", fname, df))
                continue
            except Exception:
                pass
        # image or unknown - store raw
        datasets.append(("raw", fname, content))
    return datasets

def df_sample_for_plot(df):
    if len(df) > MAX_ROWS_FOR_PLOT:
        return df.sample(MAX_ROWS_FOR_PLOT, random_state=1)
    return df

def compress_image_bytes(img_bytes, max_bytes=PLOT_MAX_BYTES):
    """
    Try to compress image bytes (PIL) until <= max_bytes.
    Returns (mime, b64uri).
    """
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        # can't open as image
        return None, None

    # try WEBP first with quality loop
    for q in (80, 60, 40, 30):
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=q, method=6)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            uri = "data:image/webp;base64," + base64.b64encode(b).decode("ascii")
            return "image/webp", uri

    # fallback to JPEG
    for q in (85, 70, 50, 30):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            uri = "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")
            return "image/jpeg", uri

    # last attempt: resize
    w, h = img.size
    for scale in (0.9, 0.8, 0.7, 0.5):
        new = img.resize((int(w*scale), int(h*scale)))
        buf = BytesIO()
        new.save(buf, format="WEBP", quality=40, method=6)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            uri = "data:image/webp;base64," + base64.b64encode(b).decode("ascii")
            return "image/webp", uri

    # give up
    return None, None

def fig_to_b64(fig, max_bytes=PLOT_MAX_BYTES):
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches='tight')
    buf.seek(0)
    b = buf.getvalue()
    # if small already
    if len(b) <= max_bytes:
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    # try compress with PIL
    mime, uri = compress_image_bytes(b, max_bytes=max_bytes)
    if uri:
        return uri
    # try lowering DPI and PNG again
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=80, bbox_inches='tight')
    b = buf.getvalue()
    if len(b) <= max_bytes:
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    # final try: convert to webp from image bytes
    mime, uri = compress_image_bytes(b, max_bytes=max_bytes)
    if uri:
        return uri
    raise ValueError("Unable to compress plot under {} bytes".format(max_bytes))

# ---------- LLM Plan Generator ----------
def plan_from_llm(question_text, available_tables):
    """
    If OpenAI API available, ask it to output a structured plan.
    The plan is a JSON with steps like:
    [{"action": "load_table", "table": "table0"}, {"action":"filter","expr":"Year<2000"}, {"action":"count"} ...]
    We keep the prompt small & structured. If LLM not available, return None.
    """
    if not LLM_OK:
        return None
    prompt = (
        "You are a data analysis planning assistant. "
        "Input: a user's question and a list of available tables (names + columns).\n"
        "Output: a JSON array of steps (no explanation). Each step is an object with "
        "action in [load, filter, compute, groupby, aggregate, regression, plot, sql]. "
        "For plot steps include x, y, kind(scatter/bar/line), regression(boolean), label_x,label_y.\n\n"
        "Respond ONLY with JSON array.\n\n"
        "QUESTION:\n" + question_text + "\n\n"
        "TABLES:\n"
    )
    for i, (name, cols) in enumerate(available_tables.items()):
        prompt += f"{name}: {cols}\n"
    try:
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0
        )
        text = resp.choices[0].text.strip()
        # try to extract JSON
        j = json.loads(text)
        return j
    except Exception:
        return None

# ---------- Executor: limited primitive actions ----------
def execute_plan(plan, tables):
    """
    Execute a plan produced by LLM (or a simple fallback plan).
    tables: dict name -> DataFrame
    Returns a list or dict of answers.
    Only supports safe primitive operations.
    """
    results = []
    context = {"tables": tables}
    for step in plan:
        action = step.get("action")
        if action == "load":
            # load a named table (it's already in tables)
            name = step.get("table")
            context["current"] = context["tables"].get(name)
        elif action == "sql":
            # run a SQL query using duckdb if available
            q = step.get("query")
            if not DUCKDB_OK:
                raise RuntimeError("DuckDB not available for SQL execution")
            res = duckdb.query(q).to_df()
            context["current"] = res
        elif action == "filter":
            expr = step.get("expr")
            # only support simple pandas query expr
            cur = context.get("current")
            if cur is None:
                raise RuntimeError("No table loaded for filter")
            cur2 = cur.query(expr)
            context["current"] = cur2
        elif action == "count":
            cur = context.get("current")
            results.append(int(len(cur)) if cur is not None else 0)
        elif action == "earliest":
            cur = context.get("current")
            col = step.get("column")
            cur[col] = pd.to_datetime(cur[col], errors='coerce')
            row = cur.loc[cur[col].idxmin()]
            results.append(row.to_dict())
        elif action == "regression":
            # return slope & intercept
            cur = context.get("current")
            x = cur[step["x"]].astype(float)
            y = cur[step["y"]].astype(float)
            mask = (~x.isna()) & (~y.isna())
            if mask.sum() < 2:
                results.append(None)
            else:
                m, b = np.polyfit(x[mask], y[mask], 1)
                results.append({"slope": float(m), "intercept": float(b)})
        elif action == "corr":
            cur = context.get("current")
            c = cur[step["x"]].corr(cur[step["y"]])
            results.append(float(c))
        elif action == "plot":
            cur = context.get("current")
            x = step["x"]
            y = step["y"]
            kind = step.get("kind", "scatter")
            # sample if necessary
            dfp = df_sample_for_plot(cur[[x, y]].dropna())
            fig, ax = plt.subplots()
            if kind == "scatter":
                ax.scatter(dfp[x], dfp[y], s=8)
            elif kind == "line":
                ax.plot(dfp[x], dfp[y])
            elif kind == "bar":
                ax.bar(dfp[x], dfp[y])
            ax.set_xlabel(step.get("label_x", x))
            ax.set_ylabel(step.get("label_y", y))
            if step.get("regression"):
                xx = dfp[x].astype(float)
                yy = dfp[y].astype(float)
                mask = (~xx.isna()) & (~yy.isna())
                if mask.sum() >= 2:
                    m, b = np.polyfit(xx[mask], yy[mask], 1)
                    ax.plot(xx, m*xx + b, 'r:' if step.get("dotted", True) else 'r-')
            uri = fig_to_b64(fig)
            results.append(uri)
        elif action == "aggregate":
            cur = context.get("current")
            gb = cur.groupby(step["by"])
            agg = gb.agg(step["agg"]).reset_index()
            results.append(agg.to_dict(orient="records"))
        else:
            # unknown action -> ignore
            continue
    return results

# ---------- Simple fallback parser (if LLM not available) ----------
def fallback_plan_for_question(question_text, tables):
    """
    Create a minimal plan for common patterns (counts, earliest, correlation, scatterplot).
    This is intentionally generic: it looks for column-like words and requests.
    """
    q = question_text.lower()
    # pick first table as table0
    table_names = list(tables.keys())
    if not table_names:
        return []
    tab0 = table_names[0]
    cols = list(tables[tab0].columns)
    plan = [{"action": "load", "table": tab0}]
    # count pattern
    if re.search(r'how many|count of|number of', q):
        plan.append({"action": "count"})
    # earliest / latest with year/date
    if re.search(r'earliest|earliest film|first', q) and any('year' in c.lower() or 'date' in c.lower() for c in cols):
        # pick year/date column
        col = next((c for c in cols if 'year' in c.lower()), next((c for c in cols if 'date' in c.lower()), cols[0]))
        plan.append({"action": "earliest", "column": col})
    # correlation
    if 'correlation' in q and len(cols) >= 2:
        plan.append({"action": "corr", "x": cols[0], "y": cols[1]})
    # scatterplot
    if 'scatter' in q or 'plot' in q or 'scatterplot' in q:
        # choose plausible numeric columns
        numcols = [c for c in cols if pd.api.types.is_numeric_dtype(tables[tab0][c])]
        if len(numcols) >= 2:
            plan.append({"action": "plot", "x": numcols[0], "y": numcols[1], "kind": "scatter", "regression": True, "dotted": True})
    return plan

# ---------- API Route ----------
@app.route("/api/", methods=["POST"])
def api_handler():
    start = now_ts()
    try:
        # 1) read questions.txt (may be in files form or raw body)
        question_text = ""
        if 'questions.txt' in request.files:
            question_text = request.files['questions.txt'].read().decode('utf-8', errors='ignore')
        else:
            # some graders POST the file as raw body
            question_text = request.get_data(as_text=True) or ""
        if not question_text:
            return jsonify({"error": "questions.txt is required"}), 400

        # 2) load attachments
        attachments = load_attachments(request.files)

        # 3) detect URLs in text
        urls = extract_urls(question_text)

        # 4) fetch remote tables (first few urls only)
        tables = {}
        table_count = 0
        for u in urls:
            try:
                obj = fetch_url_table(u)
                # if list of tables, take first and name distinct
                if isinstance(obj, list):
                    for t in obj:
                        name = f"table{table_count}"
                        tables[name] = t.head(MAX_DATA_ROWS)
                        table_count += 1
                elif isinstance(obj, pd.DataFrame):
                    name = f"table{table_count}"
                    tables[name] = obj.head(MAX_DATA_ROWS)
                    table_count += 1
                else:
                    # text - store raw
                    tables[f"text{table_count}"] = obj
                    table_count += 1
            except Exception as e:
                # skip failing URLs but continue
                tables[f"error_{table_count}"] = f"failed to fetch: {str(e)}"
                table_count += 1
            if table_count >= 5:
                break  # don't fetch endless URLs

        # 5) load attached dataframes into tables as well
        for typ, fname, content in attachments:
            if typ == "file" and isinstance(content, pd.DataFrame):
                name = f"file_{fname}"
                tables[name] = content.head(MAX_DATA_ROWS)
            else:
                tables[f"raw_{fname}"] = content

        # 6) ask LLM for plan or fallback
        available_tables = {n: list(tables[n].columns) if isinstance(tables[n], pd.DataFrame) else None for n in tables}
        plan = None
        if LLM_OK:
            plan = plan_from_llm(question_text, available_tables)
        if not plan:
            # fallback simple plan
            plan = fallback_plan_for_question(question_text, tables)

        # 7) execute plan
        answers = execute_plan(plan, tables)

        # 8) Format output depending on question requested structure
        # If question explicitly asks for JSON object (looks for braces)
        if re.search(r'\{\s*".+?"\s*:', question_text):
            # expects object mapping prompts->answers. We'll try to map if count matches.
            if isinstance(answers, list) and len(answers) == 1 and isinstance(answers[0], dict):
                out = answers[0]
            else:
                # create a best-effort mapping
                out = {}
                for i, a in enumerate(answers):
                    out[f"answer_{i+1}"] = a
            return jsonify(out)

        # If question explicitly requests JSON array (like sample), return array
        if re.search(r'json array', question_text.lower()) or re.search(r'\[\s*\d+', question_text):
            # Ensure it's an array - if answer is dict or value, wrap appropriately
            if isinstance(answers, list):
                return jsonify(answers)
            return jsonify([answers])

        # Default: return answers as array if list, else wrap
        if isinstance(answers, list):
            return jsonify(answers)
        return jsonify([answers])

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500
    finally:
        # enforce global timeout approx (handler should finish quickly)
        elapsed = now_ts() - start
        # nothing else; Render will kill long requests if needed

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
