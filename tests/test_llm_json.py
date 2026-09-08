"""The JSON layer sees real model output, which is not always clean JSON."""

from factlayer.llm import parse_json_loose


def test_plain():
    assert parse_json_loose('[{"a": 1}]') == [{"a": 1}]


def test_fenced():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}


def test_prose_padding():
    text = 'Here are the facts I found:\n[{"a": 1}]\nHope that helps!'
    assert parse_json_loose(text) == [{"a": 1}]


def test_trailing_comma():
    assert parse_json_loose('{"a": 1,}') == {"a": 1}


def test_truncated_array_keeps_complete_elements():
    """A max_tokens cut mid-array should not throw away the elements that did
    arrive -- on a 300-page document that is a lot of lost work."""
    truncated = '[{"a": 1}, {"b": 2}, {"c": '
    assert parse_json_loose(truncated) == [{"a": 1}, {"b": 2}]


def test_unparseable_returns_none():
    assert parse_json_loose("I could not find any facts.") is None
    assert parse_json_loose("") is None
