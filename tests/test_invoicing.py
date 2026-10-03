from datetime import date

import pandas as pd
import pytest
from modules.invoicing import assign_client, build_invoices, draft_cover_note

CLIENTS = pd.DataFrame({"client": ["Acme", "Globex"], "contact": ["a@x", "g@x"],
                        "tax_rate": [0.08, 0.0], "terms_days": [30, 15]})
EMPS = pd.DataFrame({"employee": ["A", "B"], "hourly_rate": [40, 28], "client_rate": [85.0, 60.0]})
TS = pd.DataFrame({
    "employee": ["A", "A", "B", "B"], "project": ["Acme web", "Globex data", "Acme API", "Mystery"],
    "hours": [8.0, 8.0, 7.5, 1.0], "flags": ["", "", "", ""]})


def test_assign_client():
    assert assign_client("acme website", CLIENTS) == "Acme"
    assert assign_client("Initech", CLIENTS) == "Unassigned"


def test_build_invoices_math():
    invs = {i["client"]: i for i in build_invoices(TS, EMPS, CLIENTS, date(2026, 10, 3))}
    acme = invs["Acme"]
    assert acme["subtotal"] == 8 * 85 + 7.5 * 60 == 1130.0
    assert acme["tax"] == 90.40 and acme["total"] == 1220.40
    assert acme["due"] == "2026-11-02"
    assert invs["Globex"]["total"] == 680.0 and invs["Globex"]["due"] == "2026-10-18"
    assert invs["Unassigned"]["warnings"]


def test_cover_note_fallback(monkeypatch):
    monkeypatch.setattr("llm.ask", lambda *a, **k: "no marker")
    inv = build_invoices(TS, EMPS, CLIENTS, date(2026, 10, 3))[0]
    assert inv["number"] in draft_cover_note(inv)


@pytest.mark.live
def test_live_cover_note():
    inv = [i for i in build_invoices(TS, EMPS, CLIENTS, date(2026, 10, 3)) if i["client"] == "Acme"][0]
    out = draft_cover_note(inv)
    print(out)
    assert "1220.4" in out and len(out) <= 700 and "I can produce" not in out
