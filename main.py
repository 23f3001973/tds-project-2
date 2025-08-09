#!/usr/bin/env python3
"""
Data Analyst Agent - universal API endpoint
Handles:
 - File uploads (.csv, .json, .parquet, images)
 - URLs inside questions.txt (scrapes CSV, JSON, HTML tables, parquet, raw text)
 - Date/numeric conversion for correlations/plots
 - Fallback to simple plan if LLM not available
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

try:
    import duckdb
    DUCKDB_OK = True
except Exception:
    DUCKDB_OK = False

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

MAX_ROWS_FOR_PLOT = 2000
PLOT_MAX_BYTES = 100_000
REQUEST_TIMEOUT = 40
MAX_DATA_ROWS = 200_000

def now_ts():
    return int(time.time())

def extract_urls(text):
    return re.findall(r'(https?://[^\s,]+|s3://[^\s,]+)', text)

def safe_read_csv_bytes(b):
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(BytesIO(b), encoding=enc)
        except Exception:
            pass
    raise

def try_read_json_bytes(b):
    try:
        return pd.read_json(BytesIO(b))
    except Exception:
        try:
            return pd.read_json(BytesIO(b), lines=True)
        except Exception:
            raise

def fetch_url_table(url):
    if re.search(r'\.csv($|\?)', url, re.I):
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return safe_read_csv_bytes(r.content)
    if re.search(r'\.json($|\?)', url, re.I):
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return try_read_json_bytes(r.content)
    if (url.endswith(".parquet") or url.startswith("s3://")) and DUCKDB_OK:
        try:
            return duckdb.query(f"SELECT * FROM read_parquet('{url}')").to_df()
        except Exception:
            pass
    try:
        tables = pd.read_html(url, flavor='bs4')
        if tables:
            return tables[0] if len(tables) == 1 else tables
    except Exception:
        pass
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def load_attachments(files):
    datasets = []
    for key in files:
        if key.lower() == "questions.txt":
            continue
        file = files.get(key)
        fname = getattr(file, "filename", key)
        content = file.read()
        if fname.lower().endswith(".csv"):
            try:
                datasets.append(("file", fname, safe_read_csv_bytes(content)))
                continue
            except Exception:
                pass
        if fname.lower().endswith((".json", ".ndjson", ".jsonl")):
            try:
                datasets.append(("file", fname, try_read_json_bytes(content)))
                continue
            except Exception:
                pass
        if fname.lower().endswith(".parquet"):
            try:
                datasets.append(("file", fname, pd.read_parquet(BytesIO(content))))
                continue
            except Exception:
                pass
        datasets.append(("raw", fname, content))
    return datasets

def df_sample_for_plot(df):
    return df.sample(MAX_ROWS_FOR_PLOT, random_state=1) if len(df) > MAX_ROWS_FOR_PLOT else df

def compress_image_bytes(img_bytes, max_bytes=PLOT_MAX_BYTES):
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None, None
    for fmt, quality in [("WEBP", 80), ("JPEG", 85)]:
        buf = BytesIO()
        img.save(buf, format=fmt, quality=quality, optimize=True)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            uri = f"data:image/{fmt.lower()};base64," + base64.b64encode(b).decode("ascii")
            return f"image/{fmt.lower()}", uri
    return None, None

def fig_to_b64(fig, max_bytes=PLOT_MAX_BYTES):
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches='tight')
    b = buf.getvalue()
    if len(b) <= max_bytes:
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    mime, uri = compress_image_bytes(b, max_bytes=max_bytes)
    if uri:
        return uri
    raise ValueError("Unable to compress plot under limit")

def convert_column_to_numeric(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series

def execute_plan(plan, tables):
    results = []
    context = {"tables": tables}
    for step in plan:
        action = step.get("action")
        if action == "load":
            context["current"] = context["tables"].get(step.get("table"))
        elif action == "corr":
            cur = context.get("current")
            x = convert_column_to_numeric(pd.to_datetime(cur[step["x"]], errors='ignore'))
            y = convert_column_to_numeric(pd.to_datetime(cur[step["y"]], errors='ignore'))
            results.append(float(pd.Series(x).corr(pd.Series(y))))
        elif action == "regression":
            cur = context.get("current")
            x = convert_column_to_numeric(cur[step["x"]])
            y = convert_column_to_numeric(cur[step["y"]])
            mask = (~x.isna()) & (~y.isna())
            if mask.sum() >= 2:
                m, b = np.polyfit(x[mask], y[mask], 1)
                results.append({"slope": float(m), "intercept": float(b)})
            else:
                results.append(None)
        elif action == "plot":
            cur = context.get("current")
            x, y = step["x"], step["y"]
            dfp = df_sample_for_plot(cur[[x, y]].dropna())
            dfp[x] = convert_column_to_numeric(dfp[x])
            dfp[y] = convert_column_to_numeric(dfp[y])
            fig, ax = plt.subplots()
            kind = step.get("kind", "scatter")
            if kind == "scatter":
                ax.scatter(dfp[x], dfp[y], s=8)
            elif kind == "line":
                ax.plot(dfp[x], dfp[y], color='red')
            elif kind == "bar":
                ax.bar(dfp[x], dfp[y], color='blue')
            ax.set_xlabel(step.get("label_x", x))
            ax.set_ylabel(step.get("label_y", y))
            results.append(fig_to_b64(fig))
        else:
            results.append(None)
    return results

def fallback_plan_for_question(question_text, tables):
    q = question_text.lower()
    if not tables:
        return []
    tab0 = list(tables.keys())[0]
    cols = list(tables[tab0].columns)
    plan = [{"action": "load", "table": tab0}]
    if 'correlation' in q and len(cols) >= 2:
        plan.append({"action": "corr", "x": cols[0], "y": cols[1]})
    if 'plot' in q:
        numcols = [c for c in cols if pd.api.types.is_numeric_dtype(tables[tab0][c]) or 'date' in c.lower()]
        if len(numcols) >= 2:
            plan.append({"action": "plot", "x": numcols[0], "y": numcols[1], "kind": "scatter"})
    return plan

@app.route("/api/", methods=["POST"])
def api_handler():
    try:
        question_text = ""
        if 'questions.txt' in request.files:
            question_text = request.files['questions.txt'].read().decode('utf-8', errors='ignore')
        else:
            question_text = request.get_data(as_text=True) or ""
        if not question_text:
            return jsonify({"error": "questions.txt is required"}), 400
        attachments = load_attachments(request.files)
        urls = extract_urls(question_text)
        tables = {}
        table_count = 0
        for u in urls[:5]:
            try:
                obj = fetch_url_table(u)
                if isinstance(obj, list):
                    for t in obj:
                        tables[f"table{table_count}"] = t.head(MAX_DATA_ROWS)
                        table_count += 1
                elif isinstance(obj, pd.DataFrame):
                    tables[f"table{table_count}"] = obj.head(MAX_DATA_ROWS)
                    table_count += 1
                else:
                    tables[f"text{table_count}"] = obj
                    table_count += 1
            except Exception as e:
                tables[f"error_{table_count}"] = str(e)
                table_count += 1
        for typ, fname, content in attachments:
            if typ == "file" and isinstance(content, pd.DataFrame):
                tables[f"file_{fname}"] = content.head(MAX_DATA_ROWS)
            else:
                tables[f"raw_{fname}"] = content
        plan = fallback_plan_for_question(question_text, tables)
        answers = execute_plan(plan, tables)
        return jsonify(answers)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
