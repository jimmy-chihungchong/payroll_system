# AI Agents for Payroll, Timesheets, Invoicing & Financial Data (Prototype)

A scoped-down, runnable prototype proving the architecture: **LLM for messy language work,
deterministic Python for money.** Runs locally on LM Studio; switch to NVIDIA NIM or Claude via `.env`.

## Architecture

```
Streamlit app.py / CLI run_demo.py
            |
        pipeline.py            one end-to-end workflow
            |
  modules/ timesheets -> payroll -> invoicing -> finance
            |
        llm.py                 single gateway: lmstudio | nim | claude (OpenAI-compatible)
```

| Module | LLM does | Python does |
|---|---|---|
| `timesheets` | Parses ambiguous free-text lines ("9 to 6, 1 hour break") | Rules parse clean lines first; hours math; flags (>12h day, weekend, duplicates) |
| `payroll` | Plain-English pay-stub explanation | Decimal math: overtime (1.5x over 40h/wk), flat tax, net pay |
| `invoicing` | Cover note per invoice | Client assignment, billing at client rate, tax, due dates |
| `finance` | Categorizes merchants the rules don't know | Aggregation; invoice-to-payment reconciliation (paid / partial / unpaid) |

### Design decisions (reliability)
- **Rules first, LLM for the exceptions.** Cheaper, faster and auditable (`source` column: `rules` / `llm`).
- **The LLM never does arithmetic.** It only receives final figures to explain.
- **Every LLM output is validated** (pydantic for JSON; length/truncation guards for text) with a **deterministic fallback**, so a bad model reply degrades gracefully instead of corrupting data.
- **Reasoning-model handling:** strips `<think>` blocks, extracts the last valid JSON from inline reasoning, and uses a `FINAL:` marker for free text.

## Setup (Windows 11, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Load a model in LM Studio and start its local server (`localhost:1234`). Set `LMSTUDIO_MODEL` in `.env`
to the exact identifier (tested with `microsoft/phi-4-reasoning-plus`).

## Run

```powershell
python run_demo.py --fast     # rules only, instant, no LLM
python run_demo.py            # full workflow with LLM (reasoning models take minutes)
streamlit run app.py          # UI; untick "Use LLM" in the sidebar for instant results
pytest -m "not live"          # fast unit tests
pytest -m live                # tests that call the LLM (slow)
```

## Switching providers (`.env`)

| Provider | Set |
|---|---|
| LM Studio (default) | `LLM_PROVIDER=lmstudio` |
| NVIDIA NIM (Streamlit deploy) | `LLM_PROVIDER=nim`, `NVIDIA_API_KEY=...`, `NIM_MODEL=...` |
| Claude | `LLM_PROVIDER=claude`, `ANTHROPIC_API_KEY=...` |

NIM model availability varies by account (e.g. `meta/llama-3.1-70b-instruct` is end-of-life); the default
`meta/llama-3.2-90b-vision-instruct` was verified for text, JSON and free-text calls.

For Streamlit Community Cloud, paste `.streamlit/secrets.toml.example` (with your key) into the app's
Secrets box; `app.py` exports them as environment variables.
Note: a hosted deployment cannot reach your local LM Studio; use `nim` or `claude` there.

## Sample data (`data/`)
`timesheet_raw.txt`, `employees.csv`, `clients.csv`, `transactions.csv`. Alice's Thursday (14h) is
flagged; INV-1001 reconciles as paid, INV-1002 as partial; "ZETA CLOUD HOSTING" needs the LLM to categorize.

## Prototype assumptions / out of scope
- Flat 20% tax and 1.5x weekly overtime are illustrative, not real tax law.
- Sample CSV/text input only. Production would add adapters (QuickBooks, Xero, Gusto, bank feeds),
  scheduling for recurring runs, approval steps, audit logging and auth.
- Claude's OpenAI-compatible endpoint is used for the `claude` provider; not tested here (no key supplied).
