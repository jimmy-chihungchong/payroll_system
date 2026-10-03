"""Payroll: all money math is deterministic Decimal; the LLM only explains the result."""
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

import llm

OT_THRESHOLD = Decimal("40")   # weekly hours before overtime
OT_MULTIPLIER = Decimal("1.5")
TAX_RATE = Decimal("0.20")     # flat, illustrative only - not real tax law


def _money(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def compute_pay(hours: float, hourly_rate: float) -> dict:
    h, rate = Decimal(str(hours)), Decimal(str(hourly_rate))
    regular_h = min(h, OT_THRESHOLD)
    ot_h = max(h - OT_THRESHOLD, Decimal(0))
    gross = _money(regular_h * rate + ot_h * rate * OT_MULTIPLIER)
    tax = _money(gross * TAX_RATE)
    return {"regular_hours": float(regular_h), "overtime_hours": float(ot_h),
            "gross": float(gross), "tax": float(tax), "net": float(gross - tax)}


def run_payroll(timesheet: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    """One pay row per employee. Entries flagged non-positive are excluded from pay."""
    valid = timesheet[~timesheet["flags"].str.contains("non-positive|duplicate", regex=True)]
    totals = valid.groupby("employee", as_index=False)["hours"].sum()
    rows = []
    for _, r in totals.merge(employees, on="employee").iterrows():
        rows.append({"employee": r["employee"], "hourly_rate": r["hourly_rate"],
                     "hours": round(r["hours"], 2), **compute_pay(r["hours"], r["hourly_rate"])})
    return pd.DataFrame(rows)


def explain_stub(row: pd.Series, flags: list[str] | None = None) -> str:
    fallback = (f"{row['employee']} worked {row['hours']}h at ${row['hourly_rate']:.2f}/h "
                f"({row['overtime_hours']}h overtime): gross ${row['gross']:.2f}, "
                f"tax ${row['tax']:.2f}, net ${row['net']:.2f}.")
    prompt = ("Write a 2-sentence plain-English explanation of this pay stub for the employee. "
              "Use ONLY these figures; do not recalculate or invent numbers.\n"
              f"{row.to_dict()}\nTax is a flat {TAX_RATE:.0%}. Overtime is paid at {OT_MULTIPLIER}x over {OT_THRESHOLD}h/week."
              + (f"\nTimesheet notes: {'; '.join(flags)}" if flags else ""))
    return llm.ask_final(prompt, fallback)
