"""Deterministic first-pass redaction for public demos and preprocessing.

This module intentionally uses only the Python standard library so that the
privacy guard can run before optional NLP/LLM dependencies are loaded.
"""
from __future__ import annotations

import re

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[ANON_PROCESS]", re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")),
    ("[ANON_CPF]", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("[ANON_CNPJ]", re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")),
    ("[ANON_EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("[ANON_PHONE]", re.compile(r"(?<!\d)(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}[-\s]?\d{4}(?!\d)")),
    ("[ANON_DATE]", re.compile(r"\b\d{2}/\d{2}/\d{4}\b")),
)


def redact_deterministic(text: str) -> str:
    """Redact common structured identifiers without external model calls."""
    redacted = text or ""
    for replacement, pattern in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted
