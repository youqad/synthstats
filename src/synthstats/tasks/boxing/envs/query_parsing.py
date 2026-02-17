"""Shared parsing helpers for Boxing task query strings."""

from __future__ import annotations

from collections.abc import Callable


def parse_query_value[T](query: str, key: str, cast: Callable[[str], T]) -> T | None:
    """Parse `key=<value>` from a query string and cast it."""
    marker = f"{key}="
    if marker not in query:
        return None

    try:
        raw = query.split(marker, 1)[1].split()[0]
    except IndexError:
        return None

    try:
        return cast(raw)
    except (TypeError, ValueError):
        return None


def parse_query_int(query: str, key: str) -> int | None:
    return parse_query_value(query, key, int)


def parse_query_float(query: str, key: str) -> float | None:
    return parse_query_value(query, key, float)
