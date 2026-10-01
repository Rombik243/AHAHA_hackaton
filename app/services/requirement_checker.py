"""Deterministic checks; missing, malformed, and disputed evidence stay unknown."""
from __future__ import annotations

import math
import operator
import re
from datetime import date
from typing import Any

from app.schemas.domain import FactData, ParsedRequirement, RequirementEvaluation
from app.schemas.supplier import FactType

SUPPLIER_SET_TYPES = {kind.value for kind in FactType}

COMPARATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gte": operator.ge,
    "lte": operator.le,
    "gt": operator.gt,
    "lt": operator.lt,
}


def _text(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


def _typed(value: Any, kind: str) -> Any:
    if kind == "number":
        if isinstance(value, (bool, list)):
            raise ValueError("Not a scalar number")
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
                raise ValueError("Not a plain number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Number must be finite")
        return number
    if kind == "date":
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date must use YYYY-MM-DD")
        return date.fromisoformat(value)
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            mapping = {"true": True, "false": False, "да": True, "нет": False, "yes": True, "no": False}
            if _text(value) in mapping:
                return mapping[_text(value)]
        raise ValueError("Not an explicit boolean")
    if isinstance(value, list):
        if not value or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("Text list must have non-empty strings")
        return tuple(sorted(set(_text(item) for item in value)))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Text must be non-empty")
    return _text(value)


def _compare(actual: Any, expected: Any, comparison: str) -> bool:
    if comparison == "contains":
        if not isinstance(expected, str):
            raise ValueError("contains expects one text value")
        values = actual if isinstance(actual, tuple) else (actual,)
        return any(expected in value for value in values)
    if comparison == "in":
        if not isinstance(expected, tuple):
            raise ValueError("in expects a list of choices")
        values = actual if isinstance(actual, tuple) else (actual,)
        return all(value in expected for value in values)
    return COMPARATORS[comparison](actual, expected)


def check_requirements(
    requirements: list[ParsedRequirement], facts: list[FactData],
) -> list[RequirementEvaluation]:
    """Check each requirement against all facts of its fact_type.

    Scalar evidence values must agree after type-specific normalization. The six
    supplier extraction types describe an incomplete set of positive statements;
    any matching statement can confirm membership, absence remains unknown.
    No recency or
    source-authority heuristic silently resolves conflicts in this MVP. For list
    values `contains` means any item contains the requested text, and `in` means
    every stated value belongs to the allowed set.
    """
    by_type: dict[str, list[FactData]] = {}
    for fact in facts:
        by_type.setdefault(_text(fact.fact_type), []).append(fact)
    checks = []
    for index, requirement in enumerate(requirements):
        evidence = by_type.get(_text(requirement.fact_type), [])
        result = RequirementEvaluation(
            requirement_index=index,
            name=requirement.name,
            status="unknown",
            reason="Нет фактов с источником для этого требования.",
            evidence=evidence,
        )
        if not evidence:
            checks.append(result)
            continue
        try:
            if any(not fact.source_url.strip() or not fact.source_quote.strip() for fact in evidence):
                raise ValueError("Missing source URL or quote")
            if _text(requirement.fact_type) in SUPPLIER_SET_TYPES:
                if requirement.type not in {"text", "enum"} or requirement.operator not in {"eq", "contains", "in"}:
                    result.reason = "Извлечённый перечень содержит только положительные текстовые сведения."
                    checks.append(result)
                    continue
                expected = _typed(requirement.expected_value, requirement.type)
                matches = [
                    fact for fact in evidence
                    if _compare(_typed(fact.value, requirement.type), expected, requirement.operator)
                ]
                if matches:
                    result.status = "confirmed"
                    result.actual_value = matches[0].value
                    result.evidence = matches
                    result.reason = "В источнике явно указано значение, соответствующее текстовому требованию."
                else:
                    result.reason = "В извлечённом перечне нет подтверждения; полнота перечня неизвестна."
                checks.append(result)
                continue
            actuals = [_typed(fact.value, requirement.type) for fact in evidence]
            expected = _typed(requirement.expected_value, requirement.type)
            if any(actual != actuals[0] for actual in actuals[1:]):
                result.reason = "Источники сообщают разные значения; требуется уточнение."
            else:
                matched = _compare(actuals[0], expected, requirement.operator)
                result.status = "confirmed" if matched else "contradicted"
                result.actual_value = evidence[0].value
                result.reason = (
                    f"Значение из источника {evidence[0].value!r} "
                    f"{'соответствует' if matched else 'не соответствует'} условию "
                    f"{requirement.operator} {requirement.expected_value!r}."
                )
        except (TypeError, ValueError, KeyError, OverflowError):
            result.reason = "Значение или доказательство нельзя надёжно проверить автоматически."
        checks.append(result)
    return checks
