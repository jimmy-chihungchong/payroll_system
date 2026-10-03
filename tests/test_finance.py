from pathlib import Path

import pandas as pd
import pytest
from modules.finance import categorize, reconcile, rule_category, summarize

TXNS = pd.read_csv(Path("data/transactions.csv"))


def test_rules():
    assert rule_category("AWS EMEA MONTHLY", -1) == "Software & Cloud"
    assert rule_category("ACME CORP PAYMENT INV-1001", 5) == "Revenue"
    assert rule_category("ZETA CLOUD HOSTING", -60) is None


def test_categorize_without_llm_marks_unresolved():
    df = categorize(TXNS, use_llm=False)
    assert set(df.loc[df.cat_source == "unresolved", "description"]) == {"ZETA CLOUD HOSTING"}
    assert summarize(df).set_index("category").loc["Payroll", "amount"] == -2400.0


def test_reconcile():
    invs = [{"number": "INV-1001", "client": "Acme", "total": 2417.85},
            {"number": "INV-1002", "client": "Globex", "total": 2470.0},
            {"number": "INV-1003", "client": "X", "total": 50.0}]
    r = reconcile(invs, TXNS).set_index("invoice")
    assert r.loc["INV-1001", "status"] == "paid"
    assert (r.loc["INV-1002", "status"], r.loc["INV-1002", "outstanding"]) == ("partial", 470.0)
    assert r.loc["INV-1003", "status"] == "unpaid"


@pytest.mark.live
def test_live_categorize():
    df = categorize(TXNS)
    print(df[["description", "category", "cat_source"]])
    row = df[df.description == "ZETA CLOUD HOSTING"].iloc[0]
    assert row["category"] in ("Software & Cloud", "Uncategorized")  # LLM answer or safe fallback
    assert (row["cat_source"] == "llm") == (row["category"] != "Uncategorized")
