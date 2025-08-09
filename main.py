import os
import io
import re
import json
import base64
import requests
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = Flask(__name__)

# ---------- Helpers ----------
def read_any_table(file_or_url):
    """Read CSV, Excel, JSON, Parquet, or HTML table from file or URL."""
    try:
        if isinstance(file_or_url, str) and re.match(r'^https?://', file_or_url):
            # Download file
            resp = requests.get(file_or_url)
            resp.raise_for_status()
            content = io.BytesIO(resp.content)
            # Try reading as CSV, Excel, JSON, HTML
            try:
                return pd.read_csv(content)
            except:
                try:
                    return pd.read_excel(content)
                except:
                    try:
                        return pd.read_json(content)
                    except:
                        try:
                            tables = pd.read_html(resp.text)
                            if tables:
                                return tables[0]
                        except:
                            return None
        else:
            # Local file path
            ext = os.path.splitext(str(file_or_url))[1].lower()
            if ext in ['.csv', '.txt']:
                return pd.read_csv(file_or_url)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_or_url)
            elif ext in ['.json']:
                return pd.read_json(file_or_url)
            elif ext in ['.parquet']:
                return pd.read_parquet(file_or_url)
            elif ext in ['.html', '.htm']:
                return pd.read_html(file_or_url)[0]
    except Exception as e:
        print(f"Error reading table from {file_or_url}: {e}")
    return None

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ---------- Analysis ----------
def analyze_sales(df):
    result = {}
    if "sales" not in df.columns:
        return {"error": "No 'sales' column found in data"}
    
    # Ensure numeric sales
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    
    # Total sales
    result["total_sales"] = df["sales"].sum()
    
    # Median sales
    result["median_sales"] = df["sales"].median()
    
    # Top region
    if "region" in df.columns:
        result["top_region"] = df.groupby("region")["sales"].sum().idxmax()
    else:
        result["top_region"] = None
    
    # Total sales tax (10% assumed)
    result["total_sales_tax"] = round(result["total_sales"] * 0.10, 2)
    
    # Day-sales correlation (if date exists)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["day"] = df["date"].dt.day
            result["day_sales_correlation"] = df["day"].corr(df["sales"])
        except:
            result["day_sales_correlation"] = None
    else:
        result["day_sales_correlation"] = None
    
    # Bar chart by region
    if "region" in df.columns:
        fig, ax = plt.subplots()
        df.groupby("region")["sales"].sum().plot(kind="bar", color="blue", ax=ax)
        ax.set_title("Total Sales by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Sales")
        result["bar_chart"] = fig_to_base64(fig)
    
    # Cumulative sales chart
    if "date" in df.columns:
        df_sorted = df.sort_values("date")
        df_sorted["cum_sales"] = df_sorted["sales"].cumsum()
        fig, ax = plt.subplots()
        ax.plot(df_sorted["date"], df_sorted["cum_sales"], color="red")
        ax.set_title("Cumulative Sales Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Sales")
        result["cumulative_sales_chart"] = fig_to_base64(fig)
    
    return result

# ---------- API ----------
@app.route("/api/", methods=["POST"])
def api_handler():
    try:
        # Load questions.txt
        if "questions.txt" in request.files:
            q_text = request.files["questions.txt"].read().decode("utf-8", errors="ignore")
        else:
            return jsonify({"error": "questions.txt not provided"}), 400
        
        # Extract possible URLs from questions
        urls = re.findall(r'(https?://\S+)', q_text)
        
        # Load dataset
        df = None
        if "sales-data.csv" in request.files:
            df = pd.read_csv(request.files["sales-data.csv"])
        elif urls:
            for u in urls:
                df = read_any_table(u)
                if df is not None:
                    break
        
        if df is None:
            return jsonify({"error": "No usable dataset found"}), 400
        
        # Perform analysis
        results = analyze_sales(df)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
