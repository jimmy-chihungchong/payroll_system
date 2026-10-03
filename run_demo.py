"""CLI demo: python run_demo.py [--fast]   (--fast = rules only, no LLM calls)"""
import argparse
from datetime import date

import pandas as pd

import pipeline

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip LLM steps")
    args = ap.parse_args()

    res = pipeline.run(**pipeline.load_sample(), use_llm=not args.fast, invoice_date=date(2026, 10, 3))

    section("1. TIMESHEETS")
    print(res["timesheet"][["employee", "date", "start", "end", "break_minutes", "hours", "source", "flags"]])
    for l in res["skipped_lines"]:
        print(f"SKIPPED (needs LLM): {l}")
    section("2. PAYROLL")
    print(res["payroll"])
    for name, text in res["stubs"].items():
        print(f"- {name}: {text}")
    section("3. INVOICES")
    for inv in res["invoices"]:
        print(f"{inv['number']} {inv['client']} due {inv['due']}  subtotal ${inv['subtotal']:.2f} "
              f"tax ${inv['tax']:.2f} TOTAL ${inv['total']:.2f}")
        for l in inv["lines"]:
            print(f"    {l['description']}: {l['hours']}h x ${l['rate']:.2f} = ${l['amount']:.2f}")
        if inv["number"] in res["notes"]:
            print(f"    Note: {res['notes'][inv['number']]}")
    section("4. FINANCIAL DATA")
    print(res["transactions"])
    print(res["summary"].to_string(index=False))
    print("\nReconciliation:")
    print(res["reconciliation"].to_string(index=False))


if __name__ == "__main__":
    main()
