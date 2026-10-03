"""Single LLM gateway. All modules call `ask()` / `ask_json()`; provider is chosen via .env."""
import json
import os
import re
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()
T = TypeVar("T", bound=BaseModel)


def _config() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for the active provider."""
    provider = os.getenv("LLM_PROVIDER", "lmstudio").lower()
    if provider == "lmstudio":
        return (os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"), "lm-studio",
                os.getenv("LMSTUDIO_MODEL", "microsoft/phi-4-mini-reasoning"))
    if provider == "nim":
        return (os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                os.environ["NVIDIA_API_KEY"], os.getenv("NIM_MODEL", "meta/llama-3.2-90b-vision-instruct"))
    if provider == "claude":  # Anthropic's OpenAI-compatible endpoint
        return ("https://api.anthropic.com/v1/", os.environ["ANTHROPIC_API_KEY"],
                os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def clean(text: str) -> str:
    """Strip <think>...</think> blocks emitted by reasoning models (also unterminated ones)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.S)
    return text.strip()


_last_truncated = False  # set by ask(): True if the last reply hit max_tokens


def ask(prompt: str, system: str = "You are a precise financial operations assistant.",
        max_tokens: int = 1500, temperature: float = 0.1) -> str:
    global _last_truncated
    base_url, key, model = _config()
    client = OpenAI(base_url=base_url, api_key=key, timeout=600)
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    _last_truncated = resp.choices[0].finish_reason == "length"
    return clean(resp.choices[0].message.content or "")


def ask_final(prompt: str, fallback: str, max_tokens: int = 6000, max_chars: int = 700, **kw) -> str:
    """Free-text answer. Models that reason inline are told to put the answer after 'FINAL:'.
    Returns `fallback` if the marker never appears (e.g. token cap hit mid-reasoning)."""
    raw = ask(f"{prompt}\n\nThink briefly. End your reply with one line starting 'FINAL:' "
              "followed by the answer, in plain text with no markdown.", max_tokens=max_tokens, **kw)
    if _last_truncated or "FINAL:" not in raw:  # truncated replies are often reasoning loops
        return fallback
    answer = raw.rsplit("FINAL:", 1)[1].strip()
    return answer if 0 < len(answer) <= max_chars else fallback


def _extract_json(text: str):
    """Pull the first JSON object/array out of a model reply (handles ```json fences + chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    dec, found, i = json.JSONDecoder(), None, 0
    while (m := re.search(r"[\[{]", text[i:])):
        start = i + m.start()
        try:
            obj, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, (list, dict)) and obj:
            found = obj  # keep the last valid one: reasoning models restate the answer at the end
        i = end
    if found is None:
        raise ValueError("no JSON found in model output")
    return found


def ask_json(prompt: str, model_cls: Type[T], retries: int = 1, **kw) -> list[T]:
    """Ask for a JSON array of `model_cls` items; validate with pydantic; retry once with the error."""
    instruction = (f"{prompt}\n\nReply with ONLY a JSON array. Each item must match this schema:\n"
                   f"{json.dumps(model_cls.model_json_schema()['properties'])}\n"
                   "Think briefly; do not over-deliberate.")
    max_tokens = kw.get("max_tokens", 8000)  # reasoning models spend thousands of tokens before answering
    last_err = ""
    for _ in range(retries + 1):
        raw = ask(instruction + (f"\n\nYour previous reply was invalid: {last_err}. Fix it." if last_err else ""),
                  max_tokens=max_tokens)
        try:
            data = _extract_json(raw)
            data = data if isinstance(data, list) else [data]
            return [model_cls.model_validate(d) for d in data]
        except (ValueError, ValidationError) as e:
            last_err = str(e)[:300]
    raise RuntimeError(f"LLM did not return valid JSON after retries: {last_err}")
