"""Financial data: rules-first categorization (LLM for unknown merchants), summaries, reconciliation."""
import re

import pandas as pd
from pydantic import BaseModel

import llm

CATEGORIES = ["Software & Cloud", "Payroll", "Travel", "Meals", "Office", "Revenue", "Other"]
RULES = {
    "Software & Cloud": ["aws", "slack", "github", "google cloud", "azure"],
    "Payroll": ["payroll"],
    "Travel": ["uber", "lyft", "airline", "hotel"],
    "Meals": ["restaurant", "cafe", "coffee"],
    "Office": ["staples", "office"],
}


class Categorized(BaseModel):
    description: str
    category: str


def rule_category(description: str, amount: float) -> str | None:
    d = description.lower()
    if amount > 0 and re.search(r"inv-\d+|payment|remittance", d):
        return "Revenue"
    for cat, words in RULES.items():
        if any(w in d for w in words):
            return cat
    return None


def categorize(txns: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
    df = txns.copy()
    df["category"] = [rule_category(d, a) for d, a in zip(df["description"], df["amount"])]
    df["cat_source"] = df["category"].map(lambda c: "rules" if pd.notna(c) else "")
    unknown = df.loc[df["category"].isna(), "description"].tolist()
    if unknown and use_llm:
        try:
            prompt = (f"Categorize each transaction into exactly one of: {CATEGORIES}.\n"
                      "Transactions:\n" + "\n".join(unknown))
            answers = {a.description.strip().lower(): a.category for a in llm.ask_json(prompt, Categorized)}
        except RuntimeError:
            answers = {}
        for i in df.index[df["category"].isna()]:
            cat = answers.get(df.at[i, "description"].strip().lower())
            if cat in CATEGORIES:
                df.at[i, "category"], df.at[i, "cat_source"] = cat, "llm"
    df["category"] = df["category"].fillna("Uncategorized")
    df.loc[df["cat_source"] == "", "cat_source"] = "unresolved"
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("category", as_index=False)["amount"].sum().round(2)
    return out.sort_values("amount")


def reconcile(invoices: list[dict], txns: pd.DataFrame) -> pd.DataFrame:
    """Match invoices to incoming payments by invoice number in the description."""
    rows = []
    for inv in invoices:
        paid = txns[(txns["amount"] > 0) & txns["description"].str.contains(inv["number"], case=False)]
        received = round(float(paid["amount"].sum()), 2)
        diff = round(inv["total"] - received, 2)
        status = "paid" if received and diff == 0 else "partial" if received else "unpaid"
        rows.append({"invoice": inv["number"], "client": inv["client"], "billed": inv["total"],
                     "received": received, "outstanding": diff, "status": status})
    return pd.DataFrame(rows)
