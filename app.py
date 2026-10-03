"""Streamlit UI: streamlit run app.py"""
import os
from datetime import date

import pandas as pd
import streamlit as st

import llm
import pipeline

st.set_page_config(page_title="AI Finance Agents (prototype)", layout="wide")

# Streamlit Cloud secrets -> env vars (llm.py reads os.environ). No-op locally without a secrets file.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass
st.title("AI Agents for Payroll, Timesheets, Invoicing & Finance")

sample = pipeline.load_sample()
_, _, model = llm._config()

with st.sidebar:
    st.header("Settings")
    st.caption(f"Provider: **{os.getenv('LLM_PROVIDER', 'lmstudio')}**  \nModel: `{model}`")
    use_llm = st.checkbox("Use LLM", value=False,
                          help="Off = rules only (instant). Local reasoning models take ~1-2 min per call; "
                               "hosted NIM models are much faster.")
    invoice_date = st.date_input("Invoice date", value=date(2026, 10, 3))
    st.divider()
    st.caption("Money math is deterministic Python. The LLM parses messy text, "
               "categorizes unknown merchants and writes explanations.")

raw = st.text_area("Raw timesheet entries (free text)", sample["raw"], height=170)
upload = st.file_uploader("Bank transactions CSV (optional; columns: date, description, amount)", type="csv")
txns = pd.read_csv(upload) if upload else sample["txns"]

if st.button("Run workflow", type="primary"):
    with st.spinner("Running agents..."):
        try:
            st.session_state["res"] = pipeline.run(raw, sample["employees"], sample["clients"], txns,
                                                   use_llm=use_llm, invoice_date=invoice_date)
        except Exception as e:  # surface LLM/connection problems instead of a stack trace
            st.error(f"Workflow failed: {e}")

res = st.session_state.get("res")
if res:
    t1, t2, t3, t4 = st.tabs(["Timesheets", "Payroll", "Invoicing", "Financial data"])
    with t1:
        st.dataframe(res["timesheet"], width="stretch")
        flagged = res["timesheet"][res["timesheet"]["flags"] != ""]
        if len(flagged):
            st.warning(f"{len(flagged)} entry(ies) flagged for review")
        for l in res["skipped_lines"]:
            st.info(f"Needs LLM (skipped in fast mode): {l}")
    with t2:
        st.dataframe(res["payroll"], width="stretch")
        for name, text in res["stubs"].items():
            st.markdown(f"**{name}** - {text}")
        st.caption("Flat 20% tax and 1.5x overtime over 40h/week are illustrative assumptions.")
    with t3:
        for inv in res["invoices"]:
            with st.expander(f"{inv['number']} - {inv['client']} - ${inv['total']:,.2f} (due {inv['due']})", True):
                st.dataframe(pd.DataFrame(inv["lines"]), width="stretch")
                st.write(f"Subtotal ${inv['subtotal']:,.2f} | Tax ({inv['tax_rate']:.0%}) ${inv['tax']:,.2f} "
                         f"| **Total ${inv['total']:,.2f}**")
                for w in inv["warnings"]:
                    st.warning(w)
                if inv["number"] in res["notes"]:
                    st.info(res["notes"][inv["number"]])
    with t4:
        c1, c2 = st.columns(2)
        c1.subheader("Categorized transactions")
        c1.dataframe(res["transactions"], width="stretch")
        c2.subheader("Spend / income by category")
        c2.bar_chart(res["summary"].set_index("category")["amount"])
        st.subheader("Invoice reconciliation")
        st.dataframe(res["reconciliation"], width="stretch")
