"""Timesheeting: LLM parses free-text entries; Python computes hours and flags anomalies."""
import re
from datetime import date, datetime

import pandas as pd
from pydantic import BaseModel

import llm

MAX_DAILY_HOURS = 12.0


class Entry(BaseModel):
    employee: str
    date: date
    start: str          # 24h "HH:MM"
    end: str            # 24h "HH:MM"
    break_minutes: int = 0
    project: str = ""
    source: str = ""    # "rules" or "llm" - audit trail for how the row was parsed


_T = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
_RANGE = re.compile(rf"\b{_T}\s*(?:to|until|-|–)\s*{_T}\b", re.I)


def _to_24h(h: str, m: str | None, ap: str | None) -> tuple[int, int] | None:
    hour, minute = int(h), int(m or 0)
    if ap:
        hour = hour % 12 + (12 if ap.lower() == "pm" else 0)
    elif hour < 13 and not m:
        return None  # bare "9" is ambiguous
    return hour, minute


def _break_minutes(text: str) -> int:
    t = text.lower()
    if "half hour" in t or "half-hour" in t:
        return 30
    if mm := re.search(r"(\d+)\s*(?:min|m)\b", t):
        return int(mm.group(1))
    if hh := re.search(r"(\d+)\s*hour", t):
        return int(hh.group(1)) * 60
    return 0


def parse_line_rules(line: str) -> Entry | None:
    """Deterministic fast path; returns None when the line is ambiguous (-> LLM fallback)."""
    emp = re.match(r"\s*([^:]+):", line)
    d = re.search(r"\d{4}-\d{2}-\d{2}", line)
    rng = _RANGE.search(line.split(":", 1)[-1].replace(d.group(0), "") if d else line)
    if not (emp and d and rng):
        return None
    g = rng.groups()
    s, e = _to_24h(*g[0:3]), _to_24h(*g[3:6])
    if not (s and e):
        return None
    rest = line.split(d.group(0), 1)[1]
    project = rest.rsplit(",", 1)[-1].strip() if "," in rest else ""
    return Entry(employee=emp.group(1).strip(), date=date.fromisoformat(d.group(0)),
                 start=f"{s[0]:02d}:{s[1]:02d}", end=f"{e[0]:02d}:{e[1]:02d}",
                 break_minutes=_break_minutes(rest.replace(rng.group(0), "")), project=project,
                 source="rules")


def parse_entries(raw_text: str) -> list[Entry]:
    """Rules first (instant, free); LLM only for lines the rules can't resolve."""
    entries: list[Entry] = []
    for line in filter(str.strip, raw_text.splitlines()):
        if (e := parse_line_rules(line)) is not None:
            entries.append(e)
            continue
        prompt = (
            "Convert this timesheet line into one structured entry. Use 24-hour HH:MM times "
            "(e.g. 5:30pm -> 17:30; a workday '9 to 6' means 09:00 to 18:00), date as YYYY-MM-DD, "
            "break_minutes as an integer, project as the client/project named.\n\n" + line
        )
        for e in llm.ask_json(prompt, Entry):
            e.source = "llm"
            entries.append(e)
    return entries


def hours_worked(e: Entry) -> float:
    fmt = "%H:%M"
    delta = datetime.strptime(e.end, fmt) - datetime.strptime(e.start, fmt)
    return round(delta.total_seconds() / 3600 - e.break_minutes / 60, 2)


def to_frame(entries: list[Entry]) -> pd.DataFrame:
    """Entries -> DataFrame with computed hours and validation flags."""
    rows = []
    for e in entries:
        h = hours_worked(e)
        flags = []
        if h <= 0:
            flags.append("non-positive hours")
        if h > MAX_DAILY_HOURS:
            flags.append(f">{MAX_DAILY_HOURS:g}h day")
        if e.date.weekday() >= 5:
            flags.append("weekend")
        rows.append({**e.model_dump(), "hours": h, "flags": "; ".join(flags)})
    df = pd.DataFrame(rows)
    dupes = df.duplicated(["employee", "date"], keep=False)
    df.loc[dupes, "flags"] = (df.loc[dupes, "flags"] + "; duplicate day").str.strip("; ")
    return df
