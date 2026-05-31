"""Tests for the JSON logging formatter and setup."""
from __future__ import annotations

import json
import logging

from agent.logging_config import JSONFormatter


def _make_record(msg: str, level: int = logging.INFO, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_required_fields():
    formatter = JSONFormatter()
    record = _make_record("hello world")
    output = json.loads(formatter.format(record))
    assert "ts" in output
    assert output["level"] == "INFO"
    assert output["logger"] == "test.logger"
    assert output["msg"] == "hello world"


def test_json_formatter_extra_fields_included():
    formatter = JSONFormatter()
    record = _make_record("with extras", query_id="abc-123", latency_ms=42)
    output = json.loads(formatter.format(record))
    assert output["query_id"] == "abc-123"
    assert output["latency_ms"] == 42


def test_json_formatter_exception_included():
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    record = _make_record("error occurred", level=logging.ERROR)
    record.exc_info = exc_info
    output = json.loads(formatter.format(record))
    assert "exc" in output
    assert "ValueError" in output["exc"]


def test_json_formatter_output_is_single_line():
    formatter = JSONFormatter()
    record = _make_record("line one\nline two")
    output = formatter.format(record)
    assert "\n" not in output


def test_setup_logging_idempotent():
    from agent.logging_config import setup_logging
    root = logging.getLogger()
    initial_handler_count = len(root.handlers)
    setup_logging("DEBUG")
    setup_logging("DEBUG")
    assert len(root.handlers) == initial_handler_count
