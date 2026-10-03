"""End-to-end workflow shared by the CLI and the Streamlit app."""
from datetime import date
from pathlib import Path

import pandas as pd

from modules import finance, invoicing, payroll, timesheets

DATA = Path(__file__).parent / "data"


def load_sample() -> dict:
    return {"raw": (DATA / "timesheet_raw.txt").read_text(),
            "employees": pd.read_csv(DATA / "employees.csv"),
            "clients": pd.read_csv(DATA / "clients.csv"),
            "txns": pd.read_csv(DATA / "transactions.csv")}


def run(raw: str, employees: pd.DataFrame, clients: pd.DataFrame, txns: pd.DataFrame,
        use_llm: bool = True, invoice_date: date | None = None) -> dict:
    """use_llm=False: rules only (instant). Ambiguous lines/merchants are reported, not guessed."""
    invoice_date = invoice_date or date.today()
    lines = [l for l in raw.splitlines() if l.strip()]
    skipped = [] if use_llm else [l for l in lines if timesheets.parse_line_rules(l) is None]
    parse_text = raw if use_llm else "\n".join(l for l in lines if l not in skipped)

    ts = timesheets.to_frame(timesheets.parse_entries(parse_text))
    pay = payroll.run_payroll(ts, employees)
    stubs = {}
    if use_llm:
        for _, row in pay.iterrows():
            emp_flags = ts.loc[(ts.employee == row["employee"]) & (ts["flags"] != ""), "flags"].tolist()
            stubs[row["employee"]] = payroll.explain_stub(row, emp_flags)

    invoices = invoicing.build_invoices(ts, employees, clients, invoice_date)
    notes = {i["number"]: invoicing.draft_cover_note(i) for i in invoices} if use_llm else {}

    cat = finance.categorize(txns, use_llm=use_llm)
    return {"timesheet": ts, "skipped_lines": skipped, "payroll": pay, "stubs": stubs,
            "invoices": invoices, "notes": notes, "transactions": cat,
            "summary": finance.summarize(cat), "reconciliation": finance.reconcile(invoices, txns)}
