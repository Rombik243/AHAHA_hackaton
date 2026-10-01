import pytest

from app.schemas.domain import FactData, ParsedRequirement
from app.services.requirement_checker import check_requirements


def make_fact(value, fact_type="experience_years"):
    return FactData(
        fact_type=fact_type, value=value, source_url="https://demo.example/about",
        source_quote=f"Опубликованное значение: {value}",
    )


def requirement(**updates):
    values = dict(name="Опыт", fact_type="experience_years", type="number", operator="gte", expected_value=5)
    values.update(updates)
    return ParsedRequirement(**values)


@pytest.mark.parametrize("actual,status", [(8, "confirmed"), (3, "contradicted"), (5, "confirmed")])
def test_numbers(actual, status):
    result = check_requirements([requirement()], [make_fact(actual)])[0]
    assert result.status == status
    assert result.actual_value == actual
    assert result.evidence[0].source_url == "https://demo.example/about"


def test_missing_fact_is_unknown_including_unrelated_fact():
    for facts in ([], [make_fact(12, fact_type="employees_count")]):
        result = check_requirements([requirement()], facts)[0]
        assert result.status == "unknown"
        assert result.actual_value is None
        assert result.evidence == []


@pytest.mark.parametrize("value", ["8 лет", "NaN", "inf", float("nan"), True, ["8"], ""])
def test_malformed_numeric_fact_is_unknown(value):
    assert check_requirements([requirement()], [make_fact(value)])[0].status == "unknown"


def test_disputed_values_are_unknown_even_when_both_pass():
    for values in ([8, 3], [8, 12]):
        result = check_requirements([requirement()], [make_fact(value) for value in values])[0]
        assert result.status == "unknown"
        assert len(result.evidence) == 2


def test_equivalent_numeric_values_agree_and_decimal_comma_is_supported():
    result = check_requirements([requirement()], [make_fact(8), make_fact("8,0")])[0]
    assert result.status == "confirmed"


def test_one_malformed_fact_does_not_get_silently_ignored():
    result = check_requirements([requirement()], [make_fact(8), make_fact("not a number")])[0]
    assert result.status == "unknown"


@pytest.mark.parametrize("op,expected,actual,status", [
    ("eq", 5, 5, "confirmed"), ("ne", 5, 5, "contradicted"),
    ("gt", 5, 5, "contradicted"), ("lt", 5, 4, "confirmed"),
    ("lte", 5, 5, "confirmed"), ("gte", 5, 4, "contradicted"),
])
def test_numeric_comparison_operators(op, expected, actual, status):
    result = check_requirements([requirement(operator=op, expected_value=expected)], [make_fact(actual)])[0]
    assert result.status == status


@pytest.mark.parametrize("actual,status", [(True, "confirmed"), (False, "contradicted"), ("да", "confirmed"), (1, "unknown")])
def test_booleans(actual, status):
    required = requirement(type="boolean", operator="eq", expected_value=True)
    assert check_requirements([required], [make_fact(actual)])[0].status == status


@pytest.mark.parametrize("actual,status", [
    ("2027-01-01", "confirmed"), ("2024-12-31", "contradicted"),
    ("2026-02-30", "unknown"), ("01.01.2027", "unknown"),
])
def test_dates(actual, status):
    required = requirement(type="date", operator="gte", expected_value="2026-01-01")
    assert check_requirements([required], [make_fact(actual)])[0].status == status


def test_enum_membership_and_text_normalization():
    required = requirement(type="enum", operator="in", expected_value=["Производитель", "Дистрибьютор"])
    assert check_requirements([required], [make_fact(" производитель ")])[0].status == "confirmed"
    assert check_requirements([required], [make_fact("розница")])[0].status == "contradicted"


def test_contains_scans_list_and_in_requires_all_values_in_allowed_set():
    required = requirement(type="text", operator="contains", expected_value="плиты")
    assert check_requirements([required], [make_fact(["Промышленные плиты", "Холодильники"])])[0].status == "confirmed"
    membership = requirement(type="enum", operator="in", expected_value=["производитель", "дистрибьютор"])
    assert check_requirements([membership], [make_fact(["производитель", "розница"])])[0].status == "contradicted"


def test_sources_are_required_even_for_prevalidated_objects():
    broken_fact = make_fact(8).model_copy(update={"source_quote": " "})
    assert check_requirements([requirement()], [broken_fact])[0].status == "unknown"


def test_empty_requirements():
    assert check_requirements([], [make_fact(8)]) == []


@pytest.mark.parametrize("kind,operator,expected", [
    ("product", "eq", "насосы"),
    ("product", "contains", "насос"),
    ("service", "in", ["насосы", "ремонт насосов"]),
    ("certificate", "eq", "ISO 9001:2015"),
])
def test_supplier_statements_form_a_set_rather_than_conflicting_scalar_values(kind, operator, expected):
    matched = "ISO 9001:2015" if kind == "certificate" else "насосы"
    facts = [make_fact("другое значение", kind), make_fact(matched, kind)]
    required = requirement(fact_type=kind, type="text", operator=operator, expected_value=expected)
    check = check_requirements([required], facts)[0]
    assert check.status == "confirmed" and check.actual_value == matched
    assert check.evidence == [facts[1]]


def test_an_incomplete_supplier_catalog_does_not_prove_product_absence():
    required = requirement(fact_type="product", type="text", operator="eq", expected_value="насосы")
    check = check_requirements([required], [make_fact("компрессоры", "product")])[0]
    assert check.status == "unknown" and check.actual_value is None


def test_supplier_positive_facts_do_not_confirm_a_negative_or_boolean_requirement():
    facts = [make_fact("Россия", "region"), make_fact("ISO 9001:2015", "certificate")]
    negative = requirement(fact_type="region", type="text", operator="ne", expected_value="Казань")
    boolean = requirement(fact_type="certificate", type="boolean", operator="eq", expected_value=True)
    assert [check.status for check in check_requirements([negative, boolean], facts)] == ["unknown", "unknown"]
