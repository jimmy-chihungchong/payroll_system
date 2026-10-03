from datetime import date
from pathlib import Path

import pytest
from modules.timesheets import Entry, hours_worked, parse_entries, to_frame


def test_hours_and_flags():
    ok = Entry(employee="A", date=date(2026, 9, 28), start="09:00", end="17:30", break_minutes=30)
    long = Entry(employee="A", date=date(2026, 10, 1), start="07:00", end="21:00")
    assert hours_worked(ok) == 8.0
    df = to_frame([ok, long])
    assert df.loc[1, "flags"] == ">12h day"
    assert df.loc[0, "flags"] == ""


def test_rules_path():
    from modules.timesheets import parse_line_rules
    e = parse_line_rules("Alice Johnson: Mon 2026-09-28 9am to 5:30pm, 30 min lunch, Acme website redesign")
    assert (e.start, e.end, e.break_minutes, e.project) == ("09:00", "17:30", 30, "Acme website redesign")
    assert parse_line_rules("Bob: Wed 2026-09-30 worked 9 to 6 with a 1 hour break, X") is None  # ambiguous
    assert parse_line_rules("Bob: Mon 2026-09-28 10:00-18:00 half hour lunch, Acme API work").break_minutes == 30


@pytest.mark.live
def test_live_parse_sample():
    raw = Path("data/timesheet_raw.txt").read_text()
    df = to_frame(parse_entries(raw))
    print(df)
    assert len(df) == 7
    alice_mon = df[(df.employee == "Alice Johnson") & (df.date == date(2026, 9, 28))].iloc[0]
    assert alice_mon.hours == 8.0
    assert ">12h day" in df[(df.employee == "Alice Johnson") & (df.date == date(2026, 10, 1))].iloc[0]["flags"]
