from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import re
import base64
import duckdb
import os
from bs4 import BeautifulSoup

app = Flask(__name__)

# ---------- Utility Functions ----------

def download_file(url):
    """Download file from URL and try to read with pandas or BeautifulSoup."""
    try:
        # Try HTML tables first
        tables = pd.read_html(url)
        if tables:
            return tables[0]
    except Exception:
        pass

    try:
        if url.lower().endswith('.csv'):
            return pd.read_csv(url)
        elif url.lower().endswith('.json'):
            return pd.read_json(url)
        elif url.lower().endswith('.parquet'):
            return pd.read_parquet(url)
    except Exception:
        pass

    # Last resort: scrape first HTML table found
    try:
        html = requests.get(url, timeout=15).text
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if table:
            return pd.read_html(str(table))[0]
    except Exception:
        pass

    return None


def load_attached_files(files):
    """Load any CSV/JSON/Parquet from attached uploads."""
    datasets = []
    for filename, file in files.items():
        try:
            if filename.lower().endswith('.csv'):
                datasets.append(pd.read_csv(file))
            elif filename.lower().endswith('.json'):
                datasets.append(pd.read_json(file))
            elif filename.lower().endswith('.parquet'):
                datasets.append(pd.read_parquet(file))
        except Exception:
            pass
    return datasets


def compress_plot(fig):
    """Return base64-encoded image under 100KB."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.read()

    # If >100KB, try JPEG compression
    if len(img_bytes) > 100_000:
        buf = BytesIO()
        fig.savefig(buf, format='jpeg', quality=70, bbox_inches='tight', dpi=80)
        buf.seek(0)
        img_bytes = buf.read()

    # Final size check
    if len(img_bytes) > 100_000:
        raise ValueError("Plot too large after compression")

    return "data:image/png;base64," + base64.b64encode(img_bytes).decode()


def detect_task(question):
    """Very basic keyword-based task detection."""
    q = question.lower()
    task = {
        'count': 'count' in q or 'number of' in q,
        'earliest': 'earliest' in q or 'first' in q,
        'latest': 'latest' in q or 'last' in q,
        'correlation': 'correlation' in q or 'relationship' in q,
        'scatter': 'scatter' in q or 'plot' in q or 'graph' in q,
        'regression': 'regression' in q or 'trend line' in q
    }
    return task


def safe_numeric(series):
    """Convert series to numeric, ignoring errors."""
    return pd.to_numeric(series, errors='coerce')


# ---------- Main Route ----------

@app.route("/api/", methods=["POST"])
def api():
    try:
        # 1. Read questions.txt
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt missing"}), 400
        question_text = request.files['questions.txt'].read().decode('utf-8')

        # 2. Detect any URLs
        urls = re.findall(r'(https?://\S+)', question_text)
        dfs = []

        for url in urls:
            df = download_file(url)
            if df is not None:
                dfs.append(df)

        # 3. Load attached datasets
        attached_dfs = load_attached_files(request.files)
        dfs.extend(attached_dfs)

        if not dfs:
            return jsonify({"error": "No data loaded"}), 400

        # Use the first dataset for now
        df = dfs[0]

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # 4. Detect task type
        task_flags = detect_task(question_text)

        results = []

        # Example logic: count rows
        if task_flags['count']:
            results.append(int(len(df)))

        # Earliest entry (assumes a date or year column exists)
        if task_flags['earliest']:
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
            if date_cols:
                col = date_cols[0]
                df[col] = pd.to_datetime(df[col], errors='coerce')
                earliest_row = df.loc[df[col].idxmin()]
                results.append(str(earliest_row.to_dict()))
            else:
                results.append(None)

        # Correlation & scatter
        if task_flags['correlation'] or task_flags['scatter']:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                corr = df[num_cols[0]].corr(df[num_cols[1]])
                if task_flags['correlation']:
                    results.append(round(corr, 4))
                if task_flags['scatter']:
                    fig, ax = plt.subplots()
                    ax.scatter(df[num_cols[0]], df[num_cols[1]])
                    ax.set_xlabel(num_cols[0])
                    ax.set_ylabel(num_cols[1])
                    if task_flags['regression']:
                        x = df[num_cols[0]].values
                        y = df[num_cols[1]].values
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        m, b = np.polyfit(x[mask], y[mask], 1)
                        ax.plot(x, m*x + b, color='red')
                    plot_uri = compress_plot(fig)
                    plt.close(fig)
                    results.append(plot_uri)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

