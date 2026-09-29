from flask import Flask, request, redirect, url_for, render_template_string, flash
import json
import os
import re
import statistics
from datetime import datetime
import calendar

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression


# ============================================================
# PERSONAL FINANCE ADVISOR
# Single-file Flask Web Prototype
# Class 11 CBSE AI Project
# ============================================================

app = Flask(__name__)
app.jinja_env.globals.update(abs=abs)
app.secret_key = "finance-advisor-prototype-key"

DATA_FILE = os.path.join(os.path.dirname(__file__), "transactions.json")


# ============================================================
# CATEGORIES
# ============================================================

CATEGORIES = [
    "Food",
    "Travel",
    "Entertainment",
    "Education",
    "Bills",
    "Shopping",
    "Health",
    "Other"
]


DEFAULT_BUDGETS = {
    "Food": 4000,
    "Travel": 2000,
    "Entertainment": 1500,
    "Education": 2500,
    "Bills": 3000,
    "Shopping": 2500,
    "Health": 2000,
    "Other": 1500
}


# ============================================================
# TRANSACTION
# ============================================================

class Transaction:

    def __init__(self, tx_id, date, description, amount, category):
        self.tx_id = int(tx_id)
        self.date = date
        self.description = description
        self.amount = float(amount)
        self.category = category

    def to_dict(self):
        return {
            "tx_id": self.tx_id,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "category": self.category
        }

    @staticmethod
    def from_dict(data):
        return Transaction(
            data["tx_id"],
            data["date"],
            data["description"],
            data["amount"],
            data["category"]
        )


# ============================================================
# TRANSACTION MANAGER
# ============================================================

class TransactionManager:

    def __init__(self):
        self.transactions = []
        self.load()

        if not self.transactions:
            self.load_sample_data()
            self.save()

    def save(self):
        with open(DATA_FILE, "w", encoding="utf-8") as file:
            json.dump(
                [transaction.to_dict() for transaction in self.transactions],
                file,
                indent=2
            )

    def load(self):
        if not os.path.exists(DATA_FILE):
            return

        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.transactions = [
                Transaction.from_dict(item)
                for item in data
            ]

        except (json.JSONDecodeError, KeyError, TypeError):
            self.transactions = []

    def load_sample_data(self):

        today = datetime.today()

        sample = [
            (32, "Swiggy dinner", 420, "Food"),
            (30, "Uber ride", 260, "Travel"),
            (28, "Netflix subscription", 499, "Entertainment"),
            (26, "Metro recharge", 200, "Travel"),
            (24, "School notebook", 150, "Education"),
            (21, "Pizza", 380, "Food"),
            (18, "Movie ticket", 300, "Entertainment"),
            (15, "Electricity bill", 1200, "Bills"),
            (12, "Zomato lunch", 350, "Food"),
            (10, "Uber ride", 240, "Travel"),
            (8, "Swiggy dinner", 400, "Food"),
            (6, "School textbook", 650, "Education"),
            (5, "Swiggy dinner", 450, "Food"),
            (4, "Uber ride", 280, "Travel"),
            (3, "Movie ticket", 350, "Entertainment"),
            (2, "Zomato dinner", 500, "Food"),
        ]

        for i, (days_ago, description, amount, category) in enumerate(
            sample,
            start=1
        ):

            date = (
                today - pd.Timedelta(days=days_ago)
            ).strftime("%Y-%m-%d")

            self.transactions.append(
                Transaction(
                    i,
                    date,
                    description,
                    amount,
                    category
                )
            )

    def add(self, description, amount, category, date=None):

        if not date:
            date = datetime.today().strftime("%Y-%m-%d")

        new_id = max(
            [transaction.tx_id for transaction in self.transactions],
            default=0
        ) + 1

        transaction = Transaction(
            new_id,
            date,
            description,
            amount,
            category
        )

        self.transactions.append(transaction)
        self.save()

        return transaction

    def delete(self, tx_id):

        self.transactions = [
            transaction
            for transaction in self.transactions
            if transaction.tx_id != tx_id
        ]

        self.save()

    def dataframe(self):

        if not self.transactions:
            return pd.DataFrame(
                columns=[
                    "tx_id",
                    "date",
                    "description",
                    "amount",
                    "category"
                ]
            )

        df = pd.DataFrame(
            [transaction.to_dict() for transaction in self.transactions]
        )

        df["date"] = pd.to_datetime(df["date"])

        return df


# ============================================================
# MACHINE LEARNING CATEGORIZER
# ============================================================

class MLCategorizer:

    TRAINING_DATA = [

        ("swiggy dinner", "Food"),
        ("zomato lunch", "Food"),
        ("pizza", "Food"),
        ("dominos order", "Food"),
        ("restaurant bill", "Food"),
        ("dinner", "Food"),
        ("lunch", "Food"),
        ("groceries", "Food"),
        ("food order", "Food"),

        ("uber ride", "Travel"),
        ("ola cab", "Travel"),
        ("metro recharge", "Travel"),
        ("bus ticket", "Travel"),
        ("petrol", "Travel"),
        ("auto fare", "Travel"),

        ("netflix subscription", "Entertainment"),
        ("movie ticket", "Entertainment"),
        ("spotify", "Entertainment"),
        ("amazon prime", "Entertainment"),
        ("game purchase", "Entertainment"),
        ("concert ticket", "Entertainment"),

        ("school notebook", "Education"),
        ("school textbook", "Education"),
        ("tuition fee", "Education"),
        ("stationery", "Education"),
        ("online course", "Education"),
        ("exam fee", "Education"),

        ("electricity bill", "Bills"),
        ("water bill", "Bills"),
        ("mobile recharge", "Bills"),
        ("internet bill", "Bills"),
        ("wifi bill", "Bills"),

        ("bought shoes", "Shopping"),
        ("new shirt", "Shopping"),
        ("amazon order", "Shopping"),
        ("shopping mall", "Shopping"),
        ("clothes", "Shopping"),

        ("doctor", "Health"),
        ("medicine", "Health"),
        ("pharmacy", "Health"),
        ("hospital", "Health")
    ]

    def __init__(self):

        descriptions = [
            item[0]
            for item in self.TRAINING_DATA
        ]

        categories = [
            item[1]
            for item in self.TRAINING_DATA
        ]

        self.vectorizer = TfidfVectorizer()

        X = self.vectorizer.fit_transform(descriptions)

        self.model = MultinomialNB()

        self.model.fit(X, categories)

    def predict(self, description):

        X = self.vectorizer.transform(
            [description.lower()]
        )

        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        confidence = max(probabilities) * 100

        return prediction, round(confidence, 1)


# ============================================================
# SMART ENTRY PARSER
# ============================================================

class SmartEntryParser:

    @classmethod
    def parse(cls, text):

        match = re.search(
            r"\d+(?:\.\d+)?",
            text
        )

        if not match:
            return None, None

        amount = float(match.group())

        description = (
            text[:match.start()]
            + " "
            + text[match.end():]
        )

        filler = {
            "spent",
            "on",
            "rs",
            "rs.",
            "inr",
            "for",
            "bought",
            "paid"
        }

        words = [
            word
            for word in description.split()
            if word.lower() not in filler
        ]

        description = " ".join(words).strip()

        if not description:
            description = "Expense"

        return amount, description


# ============================================================
# SPENDING ANALYZER
# ============================================================

class SpendingAnalyzer:

    def __init__(self, manager):
        self.manager = manager

    def current_month(self):

        df = self.manager.dataframe()

        if df.empty:
            return df

        today = datetime.today()

        return df[
            (df["date"].dt.year == today.year)
            &
            (df["date"].dt.month == today.month)
        ]

    def current_total(self):

        df = self.current_month()

        if df.empty:
            return 0

        return float(df["amount"].sum())

    def category_breakdown(self):

        df = self.current_month()

        if df.empty:
            return {}

        return (
            df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )

    def top_category(self):

        breakdown = self.category_breakdown()

        if not breakdown:
            return "None"

        return max(
            breakdown,
            key=breakdown.get
        )

    def largest_transaction(self):

        df = self.manager.dataframe()

        if df.empty:
            return None

        return df.loc[
            df["amount"].idxmax()
        ].to_dict()

    def month_comparison(self):

        df = self.manager.dataframe()

        if df.empty:
            return {}

        today = datetime.today()

        current = df[
            (df["date"].dt.year == today.year)
            &
            (df["date"].dt.month == today.month)
        ]

        if today.month == 1:
            previous_year = today.year - 1
            previous_month = 12
        else:
            previous_year = today.year
            previous_month = today.month - 1

        previous = df[
            (df["date"].dt.year == previous_year)
            &
            (df["date"].dt.month == previous_month)
        ]

        current_group = (
            current.groupby("category")["amount"].sum()
            if not current.empty
            else pd.Series(dtype=float)
        )

        previous_group = (
            previous.groupby("category")["amount"].sum()
            if not previous.empty
            else pd.Series(dtype=float)
        )

        result = {}

        categories = set(
            current_group.index
        ) | set(
            previous_group.index
        )

        for category in categories:

            current_value = float(
                current_group.get(category, 0)
            )

            previous_value = float(
                previous_group.get(category, 0)
            )

            if previous_value == 0:

                change = 100 if current_value > 0 else 0

            else:

                change = (
                    (current_value - previous_value)
                    / previous_value
                ) * 100

            result[category] = {
                "current": round(current_value, 2),
                "previous": round(previous_value, 2),
                "change": round(change, 1)
            }

        return result


# ============================================================
# ANOMALY DETECTOR
# ============================================================

class AnomalyDetector:

    def __init__(self, manager):
        self.manager = manager

    def check(self, category, amount):

        df = self.manager.dataframe()

        history = df[
            df["category"] == category
        ]["amount"].tolist()

        if len(history) < 3:

            return {
                "unusual": False,
                "reason": "Not enough history"
            }

        mean = statistics.mean(history)

        std = statistics.pstdev(history)

        if std == 0:

            return {
                "unusual": False,
                "mean": mean,
                "std": std,
                "threshold": mean
            }

        threshold = mean + (2 * std)

        unusual = amount > threshold

        return {
            "unusual": unusual,
            "mean": round(mean, 2),
            "std": round(std, 2),
            "threshold": round(threshold, 2)
        }

    def scan(self):

        df = self.manager.dataframe()

        if df.empty:
            return []

        flagged = []

        for _, row in df.iterrows():

            category = row["category"]

            amount = float(row["amount"])

            history = df[
                (df["category"] == category)
                &
                (df["tx_id"] != row["tx_id"])
            ]["amount"].tolist()

            if len(history) < 3:
                continue

            mean = statistics.mean(history)

            std = statistics.pstdev(history)

            if std == 0:
                continue

            threshold = mean + (2 * std)

            if amount > threshold:

                flagged.append({
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "description": row["description"],
                    "amount": amount,
                    "category": category,
                    "average": round(mean, 2)
                })

        return flagged


# ============================================================
# FORECAST ENGINE
# ============================================================

class ForecastEngine:

    def __init__(self, manager):
        self.manager = manager

    def pace_forecast(self):

        df = self.manager.dataframe()

        today = datetime.today()

        current = df[
            (df["date"].dt.year == today.year)
            &
            (df["date"].dt.month == today.month)
        ]

        spent = (
            float(current["amount"].sum())
            if not current.empty
            else 0
        )

        days_elapsed = today.day

        days_in_month = calendar.monthrange(
            today.year,
            today.month
        )[1]

        if days_elapsed == 0:
            return 0

        daily_average = spent / days_elapsed

        projected = (
            daily_average
            * days_in_month
        )

        return {
            "spent": round(spent, 2),
            "daily_average": round(daily_average, 2),
            "projected": round(projected, 2),
            "days_elapsed": days_elapsed,
            "days_in_month": days_in_month
        }

    def regression_forecast(self):

        df = self.manager.dataframe()

        if df.empty:
            return None

        df["month"] = df["date"].dt.to_period("M")

        monthly = (
            df.groupby("month")["amount"]
            .sum()
            .sort_index()
        )

        if len(monthly) < 2:
            return None

        X = np.arange(
            len(monthly)
        ).reshape(-1, 1)

        y = monthly.values

        model = LinearRegression()

        model.fit(X, y)

        prediction = model.predict(
            [[len(monthly)]]
        )[0]

        return round(
            max(prediction, 0),
            2
        )


# ============================================================
# BUDGET ADVISOR
# ============================================================

class BudgetAdvisor:

    def __init__(self, manager):

        self.manager = manager

        self.budgets = DEFAULT_BUDGETS.copy()

    def status(self):

        analyzer = SpendingAnalyzer(
            self.manager
        )

        breakdown = analyzer.category_breakdown()

        result = []

        for category, limit in self.budgets.items():

            spent = breakdown.get(
                category,
                0
            )

            percentage = (
                spent / limit * 100
                if limit > 0
                else 0
            )

            if percentage >= 100:
                state = "Exceeded"

            elif percentage >= 80:
                state = "Near Limit"

            else:
                state = "Healthy"

            result.append({
                "category": category,
                "spent": round(spent, 2),
                "limit": limit,
                "percentage": round(
                    percentage,
                    1
                ),
                "state": state
            })

        return result


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

class WhatIfSimulator:

    def __init__(self, manager):
        self.manager = manager

    def simulate(self, category, percentage):

        analyzer = SpendingAnalyzer(
            self.manager
        )

        breakdown = analyzer.category_breakdown()

        current = breakdown.get(
            category,
            0
        )

        saving = (
            current
            * percentage
            / 100
        )

        new_spending = (
            current - saving
        )

        return {
            "category": category,
            "current": round(
                current,
                2
            ),
            "saving": round(
                saving,
                2
            ),
            "new": round(
                new_spending,
                2
            )
        }


# ============================================================
# INSIGHT ENGINE
# ============================================================

class InsightEngine:

    def __init__(
        self,
        analyzer,
        anomaly_detector,
        budget_advisor
    ):

        self.analyzer = analyzer
        self.anomaly_detector = anomaly_detector
        self.budget_advisor = budget_advisor

    def generate(self):

        insights = []

        comparison = (
            self.analyzer.month_comparison()
        )

        if comparison:

            biggest = max(
                comparison.items(),
                key=lambda item:
                abs(item[1]["change"])
            )

            category, data = biggest

            change = data["change"]

            if change > 0:

                insights.append(
                    f"{category} spending increased by "
                    f"{change}% compared with last month."
                )

            elif change < 0:

                insights.append(
                    f"{category} spending decreased by "
                    f"{abs(change)}% compared with last month."
                )

        anomalies = (
            self.anomaly_detector.scan()
        )

        if anomalies:

            insights.append(
                f"{len(anomalies)} unusual spending "
                f"transaction(s) detected."
            )

        budgets = (
            self.budget_advisor.status()
        )

        exceeded = [
            item
            for item in budgets
            if item["state"] == "Exceeded"
        ]

        near_limit = [
            item
            for item in budgets
            if item["state"] == "Near Limit"
        ]

        if exceeded:

            insights.append(
                f"{exceeded[0]['category']} "
                f"budget has been exceeded."
            )

        elif near_limit:

            insights.append(
                f"{near_limit[0]['category']} "
                f"is approaching its budget limit."
            )

        if not insights:

            insights.append(
                "Add more transactions to unlock stronger insights."
            )

        return insights[:4]


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

manager = TransactionManager()

categorizer = MLCategorizer()

analyzer = SpendingAnalyzer(
    manager
)

anomaly_detector = AnomalyDetector(
    manager
)

forecast_engine = ForecastEngine(
    manager
)

budget_advisor = BudgetAdvisor(
    manager
)

what_if_simulator = WhatIfSimulator(
    manager
)

insight_engine = InsightEngine(
    analyzer,
    anomaly_detector,
    budget_advisor
)


# ============================================================
# HTML TEMPLATE
# ============================================================

HTML = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Finance Advisor</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {

    background:
        radial-gradient(
            circle at top right,
            #17240f 0,
            #080b09 35%,
            #050706 70%
        );

    color: #f4f7f2;

    font-family:
        Inter,
        Arial,
        sans-serif;

    min-height: 100vh;

}


/* SIDEBAR */

.sidebar {

    position: fixed;

    left: 0;
    top: 0;

    width: 230px;

    height: 100vh;

    background: #0a0e0c;

    border-right:
        1px solid #1d281f;

    padding: 28px 18px;

}


.logo {

    font-size: 23px;

    font-weight: 800;

    color: #c6ff4a;

    margin-bottom: 35px;

}


.logo span {

    color: #f4f7f2;

}


.nav a {

    display: block;

    padding: 13px 15px;

    margin-bottom: 8px;

    border-radius: 10px;

    color: #89928a;

    text-decoration: none;

    transition: .2s;

}


.nav a:hover,
.nav a.active {

    background: #182016;

    color: #c6ff4a;

}


.nav-icon {

    margin-right: 9px;

}


/* MAIN */

.main {

    margin-left: 230px;

    padding: 35px;

    max-width: 1500px;

}


.topbar {

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    margin-bottom: 30px;

}


.title h1 {

    font-size: 32px;

    margin-bottom: 6px;

}


.subtitle {

    color: #788078;

}


.badge {

    background: #172016;

    border:
        1px solid #304020;

    color: #c6ff4a;

    padding:
        8px 13px;

    border-radius: 20px;

    font-size: 13px;

}


/* CARDS */

.grid {

    display: grid;

    grid-template-columns:
        repeat(
            4,
            minmax(0, 1fr)
        );

    gap: 17px;

}


.card {

    background:
        linear-gradient(
            145deg,
            #101510,
            #0b0f0c
        );

    border:
        1px solid #1e281f;

    border-radius: 16px;

    padding: 21px;

    box-shadow:
        0 15px 40px
        rgba(0,0,0,.18);

}


.card-label {

    color: #788078;

    font-size: 13px;

    margin-bottom: 11px;

}


.card-value {

    font-size: 28px;

    font-weight: 800;

}


.green {

    color: #c6ff4a;

}


.red {

    color: #ff7171;

}


.yellow {

    color: #ffd66b;

}


/* CONTENT GRID */

.content-grid {

    display: grid;

    grid-template-columns:
        1.4fr 1fr;

    gap: 18px;

    margin-top: 18px;

}


.card-title {

    font-size: 18px;

    font-weight: 700;

    margin-bottom: 20px;

}


/* FORM */

input,
select {

    width: 100%;

    background: #080b09;

    border:
        1px solid #273129;

    color: #f4f7f2;

    padding: 12px;

    border-radius: 9px;

    outline: none;

}


input:focus,
select:focus {

    border-color:
        #8fb83a;

}


label {

    display: block;

    color: #8c958d;

    font-size: 13px;

    margin:
        12px 0 6px;

}


button {

    border: none;

    cursor: pointer;

    background: #c6ff4a;

    color: #071006;

    font-weight: 800;

    padding:
        12px 18px;

    border-radius: 9px;

    margin-top: 14px;

}


button:hover {

    filter: brightness(1.08);

}


.secondary {

    background: #182018;

    color: #c6ff4a;

    border:
        1px solid #34422b;

}


/* BARS */

.bar-row {

    margin-bottom: 16px;

}


.bar-info {

    display: flex;

    justify-content:
        space-between;

    font-size: 13px;

    margin-bottom: 6px;

}


.bar-bg {

    height: 9px;

    background: #1b221c;

    border-radius: 20px;

    overflow: hidden;

}


.bar {

    height: 100%;

    background: #c6ff4a;

    border-radius: 20px;

}


/* INSIGHTS */

.insight {

    border-left:
        3px solid #c6ff4a;

    background: #121812;

    padding: 13px;

    margin-bottom: 10px;

    border-radius:
        0 9px 9px 0;

    color: #c4ccc4;

}


/* TABLE */

.table-wrapper {

    overflow-x: auto;

}


table {

    width: 100%;

    border-collapse:
        collapse;

}


th {

    color: #69736b;

    font-size: 12px;

    text-align: left;

    padding: 11px;

    border-bottom:
        1px solid #202920;

}


td {

    padding: 13px 11px;

    border-bottom:
        1px solid #151c16;

    font-size: 14px;

}


.category {

    color: #c6ff4a;

}


.delete {

    color: #ff7777;

    text-decoration: none;

}


/* ALERT */

.alert {

    padding: 12px 15px;

    margin-bottom: 18px;

    border-radius: 10px;

    background: #172016;

    border:
        1px solid #334228;

    color: #c6ff4a;

}


/* BUDGET */

.budget-item {

    margin-bottom: 18px;

}


.budget-header {

    display: flex;

    justify-content:
        space-between;

    margin-bottom: 7px;

    font-size: 13px;

}


.budget-bar {

    height: 8px;

    background: #1c241d;

    border-radius: 20px;

}


.budget-fill {

    height: 100%;

    background: #c6ff4a;

    border-radius: 20px;

}


.budget-fill.warning {

    background: #ffd66b;

}


.budget-fill.danger {

    background: #ff7171;

}


/* TWO FORM COLUMNS */

.form-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 12px;

}


/* FOOTER */

.footer {

    color: #59625b;

    font-size: 12px;

    text-align: center;

    margin-top: 35px;

}


/* RESPONSIVE */

@media(max-width: 1000px) {

    .sidebar {

        position: static;

        width: 100%;

        height: auto;

        border-right: none;

        border-bottom:
            1px solid #1d281f;

    }

    .main {

        margin-left: 0;

    }

    .grid {

        grid-template-columns:
            repeat(
                2,
                1fr
            );

    }

}


@media(max-width: 650px) {

    .main {

        padding: 18px;

    }

    .grid,
    .content-grid,
    .form-grid {

        grid-template-columns:
            1fr;

    }

    .topbar {

        align-items:
            flex-start;

        gap: 15px;

        flex-direction:
            column;

    }

}


</style>

</head>


<body>


<!-- SIDEBAR -->

<aside class="sidebar">

    <div class="logo">
        FINANCE<span>ADVISOR</span>
    </div>

    <nav class="nav">

        <a class="active"
           href="/">
           <span class="nav-icon">▦</span>
           Dashboard
        </a>

        <a href="/transactions">
           <span class="nav-icon">₹</span>
           Transactions
        </a>

        <a href="/analysis">
           <span class="nav-icon">◒</span>
           Analysis
        </a>

        <a href="/budget">
           <span class="nav-icon">◎</span>
           Budget
        </a>

        <a href="/forecast">
           <span class="nav-icon">↗</span>
           Forecast
        </a>

        <a href="/what-if">
           <span class="nav-icon">◇</span>
           What-If
        </a>

    </nav>

</aside>


<!-- MAIN -->

<main class="main">


<div class="topbar">

    <div class="title">

        <h1>Financial Overview</h1>

        <div class="subtitle">
            Understand your spending. Make smarter decisions.
        </div>

    </div>

    <div class="badge">
        AI Prototype
    </div>

</div>


{% with messages = get_flashed_messages() %}

    {% if messages %}

        {% for message in messages %}

            <div class="alert">
                {{ message }}
            </div>

        {% endfor %}

    {% endif %}

{% endwith %}


<!-- SUMMARY -->

<div class="grid">

    <div class="card">

        <div class="card-label">
            THIS MONTH
        </div>

        <div class="card-value green">
            ₹{{ "{:,.0f}".format(total) }}
        </div>

    </div>


    <div class="card">

        <div class="card-label">
            TOP CATEGORY
        </div>

        <div class="card-value">
            {{ top_category }}
        </div>

    </div>


    <div class="card">

        <div class="card-label">
            TRANSACTIONS
        </div>

        <div class="card-value">
            {{ transaction_count }}
        </div>

    </div>


    <div class="card">

        <div class="card-label">
            PROJECTED MONTH END
        </div>

        <div class="card-value">
            ₹{{ "{:,.0f}".format(forecast.projected) }}
        </div>

    </div>

</div>


<!-- MAIN CONTENT -->

<div class="content-grid">


<!-- SPENDING -->

<div class="card">

    <div class="card-title">
        Spending by Category
    </div>

    {% if breakdown %}

        {% set max_value =
            breakdown.values()|max %}

        {% for category, amount
              in breakdown.items() %}

            <div class="bar-row">

                <div class="bar-info">

                    <span>
                        {{ category }}
                    </span>

                    <span>
                        ₹{{ "{:,.0f}".format(amount) }}
                    </span>

                </div>

                <div class="bar-bg">

                    <div
                        class="bar"
                        style="width:
                        {{ (amount / max_value * 100)
                           if max_value else 0 }}%">
                    </div>

                </div>

            </div>

        {% endfor %}

    {% else %}

        <p class="subtitle">
            No spending data yet.
        </p>

    {% endif %}

</div>


<!-- AI INSIGHTS -->

<div class="card">

    <div class="card-title">
        🤖 AI Insights
    </div>

    {% for insight in insights %}

        <div class="insight">
            {{ insight }}
        </div>

    {% endfor %}

</div>

</div>


<!-- QUICK ADD + BUDGET -->

<div class="content-grid">


<!-- QUICK ADD -->

<div class="card">

    <div class="card-title">
        ⚡ Quick Add
    </div>

    <form method="POST"
          action="/quick-add">

        <label>
            Describe your expense naturally
        </label>

        <input
            type="text"
            name="text"
            placeholder="450 swiggy dinner"
            required
        >

        <button type="submit">
            Analyze & Add
        </button>

    </form>

    <p class="subtitle"
       style="margin-top:15px">

        Example:
        <b>₹450 Swiggy dinner</b>

    </p>

</div>


<!-- BUDGET -->

<div class="card">

    <div class="card-title">
        🎯 Budget Status
    </div>

    {% for item in budgets[:5] %}

        <div class="budget-item">

            <div class="budget-header">

                <span>
                    {{ item.category }}
                </span>

                <span>
                    ₹{{ "{:,.0f}".format(item.spent) }}
                    /
                    ₹{{ "{:,.0f}".format(item.limit) }}
                </span>

            </div>

            <div class="budget-bar">

                <div
                    class="budget-fill
                    {% if item.percentage >= 100 %}
                        danger
                    {% elif item.percentage >= 80 %}
                        warning
                    {% endif %}"
                    style="width:
                    {{ [item.percentage, 100]|min }}%">
                </div>

            </div>

        </div>

    {% endfor %}

</div>

</div>


<!-- RECENT TRANSACTIONS -->

<div class="card"
     style="margin-top:18px">

    <div class="card-title">
        Recent Transactions
    </div>

    <div class="table-wrapper">

        <table>

            <thead>

                <tr>

                    <th>
                        DATE
                    </th>

                    <th>
                        DESCRIPTION
                    </th>

                    <th>
                        CATEGORY
                    </th>

                    <th>
                        AMOUNT
                    </th>

                </tr>

            </thead>

            <tbody>

            {% for transaction
                  in transactions[:8] %}

                <tr>

                    <td>
                        {{ transaction.date }}
                    </td>

                    <td>
                        {{ transaction.description }}
                    </td>

                    <td class="category">
                        {{ transaction.category }}
                    </td>

                    <td>
                        ₹{{ "{:,.0f}".format(transaction.amount) }}
                    </td>

                </tr>

            {% endfor %}

            </tbody>

        </table>

    </div>

</div>


<div class="footer">

    Personal Finance Advisor ·
    Class 11 CBSE AI Prototype ·
    Built with Python + Flask

</div>


</main>

</body>

</html>

"""


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    total = analyzer.current_total()

    breakdown = analyzer.category_breakdown()

    top_category = analyzer.top_category()

    df = manager.dataframe()

    transactions = []

    if not df.empty:

        df = df.sort_values(
            "date",
            ascending=False
        )

        for _, row in df.iterrows():

            transactions.append({
                "id": int(row["tx_id"]),
                "date": row["date"].strftime(
                    "%Y-%m-%d"
                ),
                "description": row["description"],
                "amount": float(row["amount"]),
                "category": row["category"]
            })

    forecast = (
        forecast_engine.pace_forecast()
    )

    budgets = budget_advisor.status()

    insights = insight_engine.generate()

    return render_template_string(
        HTML,
        total=total,
        breakdown=breakdown,
        top_category=top_category,
        transaction_count=len(
            transactions
        ),
        forecast=forecast,
        budgets=budgets,
        insights=insights,
        transactions=transactions
    )


# ============================================================
# QUICK ADD
# ============================================================

@app.route(
    "/quick-add",
    methods=["POST"]
)
def quick_add():

    text = request.form.get(
        "text",
        ""
    ).strip()

    amount, description = (
        SmartEntryParser.parse(text)
    )

    if amount is None:

        flash(
            "I couldn't find an amount. Try: 450 swiggy dinner"
        )

        return redirect(
            url_for("dashboard")
        )

    category, confidence = (
        categorizer.predict(
            description
        )
    )

    transaction = manager.add(
        description,
        amount,
        category
    )

    anomaly = (
        anomaly_detector.check(
            category,
            amount
        )
    )

    if anomaly.get("unusual"):

        flash(
            f"⚠️ Unusual spending detected: "
            f"₹{amount:,.0f} in {category}. "
            f"Your historical average is "
            f"₹{anomaly['mean']:,.0f}."
        )

    else:

        flash(
            f"✓ Added ₹{amount:,.0f} "
            f"{description} → "
            f"{category} "
            f"({confidence}% AI confidence)"
        )

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# TRANSACTIONS PAGE
# ============================================================

@app.route("/transactions")
def transactions_page():

    df = manager.dataframe()

    transactions = []

    if not df.empty:

        df = df.sort_values(
            "date",
            ascending=False
        )

        for _, row in df.iterrows():

            transactions.append({
                "id": int(row["tx_id"]),
                "date": row["date"].strftime(
                    "%Y-%m-%d"
                ),
                "description": row["description"],
                "amount": float(row["amount"]),
                "category": row["category"]
            })

    return render_template_string(
        TRANSACTIONS_HTML,
        transactions=transactions
    )


# ============================================================
# DELETE
# ============================================================

@app.route(
    "/delete/<int:tx_id>"
)
def delete_transaction(tx_id):

    manager.delete(tx_id)

    flash("Transaction deleted.")

    return redirect(
        url_for("transactions_page")
    )


# ============================================================
# ANALYSIS PAGE
# ============================================================

@app.route("/analysis")
def analysis_page():

    breakdown = analyzer.category_breakdown()

    comparison = analyzer.month_comparison()

    anomalies = anomaly_detector.scan()

    largest = analyzer.largest_transaction()

    return render_template_string(
        ANALYSIS_HTML,
        breakdown=breakdown,
        comparison=comparison,
        anomalies=anomalies,
        largest=largest
    )


# ============================================================
# BUDGET PAGE
# ============================================================

@app.route("/budget")
def budget_page():

    budgets = budget_advisor.status()

    return render_template_string(
        BUDGET_HTML,
        budgets=budgets
    )


# ============================================================
# FORECAST PAGE
# ============================================================

@app.route("/forecast")
def forecast_page():

    pace = forecast_engine.pace_forecast()

    regression = (
        forecast_engine
        .regression_forecast()
    )

    return render_template_string(
        FORECAST_HTML,
        pace=pace,
        regression=regression
    )


# ============================================================
# WHAT-IF PAGE
# ============================================================

@app.route(
    "/what-if",
    methods=["GET", "POST"]
)
def what_if_page():

    result = None

    if request.method == "POST":

        category = request.form.get(
            "category"
        )

        percentage = float(
            request.form.get(
                "percentage"
            )
        )

        result = (
            what_if_simulator
            .simulate(
                category,
                percentage
            )
        )

    return render_template_string(
        WHATIF_HTML,
        categories=CATEGORIES,
        result=result
    )


# ============================================================
# SHARED PAGE STYLE
# ============================================================

PAGE_STYLE = """

<style>

body {
    background:#070907;
    color:#f4f7f2;
    font-family:Arial,sans-serif;
    margin:0;
}

.container {
    max-width:1100px;
    margin:auto;
    padding:40px 25px;
}

a {
    color:#c6ff4a;
    text-decoration:none;
}

h1 {
    margin-bottom:8px;
}

.subtitle {
    color:#7f897f;
}

.card {
    background:#101510;
    border:1px solid #202a20;
    border-radius:15px;
    padding:22px;
    margin-top:18px;
}

table {
    width:100%;
    border-collapse:collapse;
}

th,td {
    padding:13px;
    border-bottom:1px solid #202820;
    text-align:left;
}

th {
    color:#727c73;
    font-size:12px;
}

.green {
    color:#c6ff4a;
}

.red {
    color:#ff7171;
}

.yellow {
    color:#ffd66b;
}

button {
    background:#c6ff4a;
    color:#081006;
    border:0;
    padding:12px 18px;
    border-radius:8px;
    font-weight:bold;
    cursor:pointer;
}

select,input {
    background:#080b09;
    color:white;
    border:1px solid #293329;
    padding:11px;
    border-radius:8px;
    width:100%;
    margin:7px 0 15px;
}

.back {
    display:inline-block;
    margin-bottom:25px;
}

</style>

"""


# ============================================================
# TRANSACTIONS HTML
# ============================================================

TRANSACTIONS_HTML = PAGE_STYLE + """

<div class="container">

<a class="back" href="/">
← Back to Dashboard
</a>

<h1>Transactions</h1>

<p class="subtitle">
Your complete spending history.
</p>

<div class="card">

<table>

<tr>
<th>Date</th>
<th>Description</th>
<th>Category</th>
<th>Amount</th>
<th></th>
</tr>

{% for t in transactions %}

<tr>

<td>
{{ t.date }}
</td>

<td>
{{ t.description }}
</td>

<td class="green">
{{ t.category }}
</td>

<td>
₹{{ "{:,.0f}".format(t.amount) }}
</td>

<td>
<a class="red"
href="/delete/{{ t.id }}">
Delete
</a>
</td>

</tr>

{% endfor %}

</table>

</div>

</div>

"""


# ============================================================
# ANALYSIS HTML
# ============================================================

ANALYSIS_HTML = PAGE_STYLE + """

<div class="container">

<a class="back" href="/">
← Back to Dashboard
</a>

<h1>Spending Analysis</h1>

<p class="subtitle">
AI-assisted analysis of your spending patterns.
</p>


<div class="card">

<h2>Category Breakdown</h2>

{% for category, amount in breakdown.items() %}

<p>
<b>{{ category }}</b>
—
₹{{ "{:,.0f}".format(amount) }}
</p>

{% endfor %}

</div>


<div class="card">

<h2>Month-over-Month Change</h2>

<table>

<tr>
<th>Category</th>
<th>This Month</th>
<th>Last Month</th>
<th>Change</th>
</tr>

{% for category, data in comparison.items() %}

<tr>

<td>{{ category }}</td>

<td>
₹{{ "{:,.0f}".format(data.current) }}
</td>

<td>
₹{{ "{:,.0f}".format(data.previous) }}
</td>

<td class="
{% if data.change > 0 %}
red
{% else %}
green
{% endif %}
">

{% if data.change > 0 %}
↑
{% else %}
↓
{% endif %}

{{ abs(data.change) }}%

</td>

</tr>

{% endfor %}

</table>

</div>


<div class="card">

<h2>🚨 Unusual Spending</h2>

{% if anomalies %}

{% for item in anomalies %}

<p class="red">

₹{{ "{:,.0f}".format(item.amount) }}
—
{{ item.description }}
—
{{ item.category }}

</p>

{% endfor %}

{% else %}

<p class="green">
No unusual transactions detected.
</p>

{% endif %}

</div>


{% if largest %}

<div class="card">

<h2>Largest Expense</h2>

<p>
<b>{{ largest.description }}</b>
</p>

<p class="green">
₹{{ "{:,.0f}".format(largest.amount) }}
</p>

</div>

{% endif %}


</div>

"""


# ============================================================
# BUDGET HTML
# ============================================================

BUDGET_HTML = PAGE_STYLE + """

<div class="container">

<a class="back" href="/">
← Back to Dashboard
</a>

<h1>Budget Advisor</h1>

<p class="subtitle">
Monitor category limits and avoid overspending.
</p>

{% for item in budgets %}

<div class="card">

<h3>
{{ item.category }}
</h3>

<p>

₹{{ "{:,.0f}".format(item.spent) }}

/

₹{{ "{:,.0f}".format(item.limit) }}

—

<span class="
{% if item.state == 'Exceeded' %}
red
{% elif item.state == 'Near Limit' %}
yellow
{% else %}
green
{% endif %}
">

{{ item.state }}

</span>

</p>

</div>

{% endfor %}

</div>

"""


# ============================================================
# FORECAST HTML
# ============================================================

FORECAST_HTML = PAGE_STYLE + """

<div class="container">

<a class="back" href="/">
← Back to Dashboard
</a>

<h1>Spending Forecast</h1>

<p class="subtitle">
Simple, explainable projections — not guaranteed predictions.
</p>


<div class="card">

<h2>Current Month Pace</h2>

<p>
Spent so far:
<b>
₹{{ "{:,.0f}".format(pace.spent) }}
</b>
</p>

<p>
Daily average:
<b>
₹{{ "{:,.0f}".format(pace.daily_average) }}
</b>
</p>

<h2 class="green">
Projected month-end:
₹{{ "{:,.0f}".format(pace.projected) }}
</h2>

</div>


<div class="card">

<h2>Linear Regression</h2>

{% if regression is not none %}

<p>
Predicted next-month spending:
</p>

<h2 class="green">
₹{{ "{:,.0f}".format(regression) }}
</h2>

<p class="subtitle">
Based on historical monthly totals.
</p>

{% else %}

<p class="subtitle">
At least two months of data are needed
for regression forecasting.
</p>

{% endif %}

</div>

</div>

"""


# ============================================================
# WHAT-IF HTML
# ============================================================

WHATIF_HTML = PAGE_STYLE + """

<div class="container">

<a class="back" href="/">
← Back to Dashboard
</a>

<h1>What-If Simulator</h1>

<p class="subtitle">
Test hypothetical changes without modifying your real data.
</p>

<div class="card">

<form method="POST">

<label>
Category
</label>

<select name="category">

{% for category in categories %}

<option>
{{ category }}
</option>

{% endfor %}

</select>


<label>
Reduce spending by (%)
</label>

<input
type="number"
name="percentage"
min="1"
max="100"
value="15"
required
>


<button>
Simulate
</button>

</form>

</div>


{% if result %}

<div class="card">

<h2>
What if {{ result.category }}
spending fell by {{ request.form.percentage }}%?
</h2>

<p>
Current spending:
<b>
₹{{ "{:,.0f}".format(result.current) }}
</b>
</p>

<p class="green">

Potential savings:
<b>
₹{{ "{:,.0f}".format(result.saving) }}
</b>

</p>

<p>
Hypothetical spending:
<b>
₹{{ "{:,.0f}".format(result.new) }}
</b>
</p>

</div>

{% endif %}

</div>

"""


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 55)
    print(" PERSONAL FINANCE ADVISOR")
    print("=" * 55)
    print()
    print(" Website running at:")
    print(f" http://127.0.0.1:{os.environ.get('PORT', 5000)}")
    print()
    print(" Press CTRL+C to stop.")
    print()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))