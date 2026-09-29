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
