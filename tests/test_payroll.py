import pandas as pd
import pytest
from modules.payroll import compute_pay, explain_stub, run_payroll


def test_no_overtime():
    p = compute_pay(37.75, 40.0)
    assert p == {"regular_hours": 37.75, "overtime_hours": 0.0, "gross": 1510.0, "tax": 302.0, "net": 1208.0}


def test_overtime():
    p = compute_pay(45, 20.0)  # 40*20 + 5*30 = 950
    assert (p["overtime_hours"], p["gross"], p["tax"], p["net"]) == (5.0, 950.0, 190.0, 760.0)


def test_run_payroll_excludes_bad_rows():
    ts = pd.DataFrame({"employee": ["A", "A", "A"], "hours": [8.0, 8.0, -1.0],
                       "flags": ["", "", "non-positive hours"]})
    emp = pd.DataFrame({"employee": ["A"], "hourly_rate": [10.0], "client_rate": [20.0]})
    assert run_payroll(ts, emp).loc[0, "gross"] == 160.0


def test_explain_fallback_when_llm_unusable(monkeypatch):
    monkeypatch.setattr("llm.ask", lambda *a, **k: "rambling with no marker")
    row = pd.Series({"employee": "A", "hours": 8, "hourly_rate": 10.0, "overtime_hours": 0.0,
                     "gross": 80.0, "tax": 16.0, "net": 64.0})
    assert "net $64.00" in explain_stub(row)


@pytest.mark.live
def test_live_explain():
    row = pd.Series({"employee": "Alice Johnson", "hours": 37.75, "hourly_rate": 40.0,
                     "overtime_hours": 0.0, "gross": 1510.0, "tax": 302.0, "net": 1208.0})
    out = explain_stub(row)
    print(out)
    assert "1208" in out and "<think>" not in out
