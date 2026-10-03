import pytest
from llm import ask, ask_json, clean, _extract_json
from pydantic import BaseModel


def test_clean_strips_think():
    assert clean("<think>hmm</think> hello") == "hello"
    assert clean("<think>never closed") == ""


def test_extract_json_fenced():
    assert _extract_json('sure!\n```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_extract_json_after_reasoning_prose():
    txt = 'Keys are {employee,date}. I will answer:\n[{"a": 1}]\nFinal:[{"a": 1}]'
    assert _extract_json(txt) == [{"a": 1}]


@pytest.mark.live
def test_live_ask():
    assert "4" in ask("What is 2+2? Answer with just the number.", max_tokens=1500)


@pytest.mark.live
def test_live_json():
    class Item(BaseModel):
        name: str
        qty: int
    items = ask_json('Extract: "3 apples and 2 pears".', Item)
    assert {i.name.lower().rstrip("s") for i in items} >= {"apple", "pear"}
