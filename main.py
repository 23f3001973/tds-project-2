import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/", methods=["POST"])
def analyze():
    # Read the questions.txt
    questions_file = request.files.get("questions.txt")
    if not questions_file:
        return jsonify({"error": "questions.txt is required"}), 400
    questions_text = questions_file.read().decode("utf-8")

    # Example: Wikipedia highest-grossing films
    if "highest grossing films" in questions_text.lower():
        # Scrape the Wikipedia table
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        tables = pd.read_html(url)
        df = tables[0]  # first table

        # 1️⃣ How many $2bn movies before 2000?
        df["Worldwide gross"] = df["Worldwide gross"].replace(r"[\$,]", "", regex=True).astype(float)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        count_2bn_before_2000 = df[(df["Worldwide gross"] >= 2_000_000_000) &
                                   (df["Year"] < 2000)].shape[0]

        # 2️⃣ Earliest film over $1.5bn
        earliest_over_1_5bn = df[df["Worldwide gross"] > 1_500_000_000].sort_values("Year").iloc[0]["Title"]

        # 3️⃣ Correlation Rank vs Peak
        if "Peak" in df.columns:
            corr = df["Rank"].corr(df["Peak"])
        else:
            corr = None

        # 4️⃣ Scatterplot with dotted red regression line
        plt.figure()
        plt.scatter(df["Rank"], df["Peak"], label="Data points")
        m, b = pd.Series(df["Peak"]).corr(df["Rank"]), 0
        plt.plot(df["Rank"], m * df["Rank"] + b, "r:", label="Regression line")
        plt.xlabel("Rank")
        plt.ylabel("Peak")
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img_uri = f"data:image/png;base64,{img_base64}"

        return jsonify([count_2bn_before_2000, earliest_over_1_5bn, corr, img_uri])

    return jsonify({"error": "Unsupported question"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
