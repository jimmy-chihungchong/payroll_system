"""Invoicing: deterministic billing math per client; the LLM drafts the cover note."""
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

import llm


def _money(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def assign_client(project: str, clients: pd.DataFrame) -> str:
    """A project belongs to the first known client whose name appears in it; else 'Unassigned'."""
    for name in clients["client"]:
        if name.lower() in project.lower():
            return name
    return "Unassigned"


def build_invoices(timesheet: pd.DataFrame, employees: pd.DataFrame, clients: pd.DataFrame,
                   invoice_date: date, first_number: int = 1001) -> list[dict]:
    ts = timesheet[~timesheet["flags"].str.contains("non-positive|duplicate", regex=True)].copy()
    ts["client"] = ts["project"].apply(lambda p: assign_client(p, clients))
    ts = ts.merge(employees[["employee", "client_rate"]], on="employee")
    invoices = []
    for n, (client, grp) in enumerate(ts.groupby("client"), start=first_number):
        crow = clients[clients["client"] == client]
        tax_rate = Decimal(str(crow["tax_rate"].iloc[0])) if len(crow) else Decimal(0)
        terms = int(crow["terms_days"].iloc[0]) if len(crow) else 30
        lines = []
        for (emp, proj, rate), g in grp.groupby(["employee", "project", "client_rate"]):
            hours = Decimal(str(round(g["hours"].sum(), 2)))
            amount = _money(hours * Decimal(str(rate)))
            lines.append({"description": f"{proj} - {emp}", "hours": float(hours),
                          "rate": float(rate), "amount": float(amount)})
        subtotal = _money(sum(Decimal(str(l["amount"])) for l in lines))
        tax = _money(subtotal * tax_rate)
        invoices.append({
            "number": f"INV-{n}", "client": client, "date": invoice_date.isoformat(),
            "due": (invoice_date + timedelta(days=terms)).isoformat(), "lines": lines,
            "subtotal": float(subtotal), "tax_rate": float(tax_rate), "tax": float(tax),
            "total": float(subtotal + tax),
            "contact": crow["contact"].iloc[0] if len(crow) else "",
            "warnings": ["No matching client record"] if client == "Unassigned" else [],
        })
    return invoices


def draft_cover_note(inv: dict) -> str:
    fallback = (f"Please find attached invoice {inv['number']} for ${inv['total']:.2f}, "
                f"due {inv['due']}. Thank you for your business.")
    items = "; ".join(f"{l['description']}: {l['hours']}h" for l in inv["lines"])
    prompt = ("Write a polite, 2-sentence cover note for this invoice to the client. "
              "Use ONLY these facts; do not change any numbers.\n"
              f"Client: {inv['client']}, Invoice {inv['number']}, total ${inv['total']:.2f}, "
              f"due {inv['due']}. Work: {items}.")
    return llm.ask_final(prompt, fallback)
