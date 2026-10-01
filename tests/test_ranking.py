import pytest

from app.config import Settings
from app.schemas.domain import CompanyCandidate, FactData, ParsedRequirement
from app.services.ranking import rank_candidates


def required():
    return [ParsedRequirement(name="Опыт", fact_type="experience", type="number", operator="gte", expected_value=5)]


def candidate(name, value=None, relevance=0.0):
    facts = [] if value is None else [FactData(
        fact_type="experience", value=value, source_url="https://demo.example/about", source_quote=f"Опыт: {value}",
    )]
    return CompanyCandidate(name=name, facts=facts, relevance=relevance)


def test_contradiction_is_worse_than_unknown_and_relevance_cannot_override_checks():
    results = rank_candidates([
        candidate("Недостаточно", 3, relevance=1),
        candidate("Неизвестно", relevance=0.7),
        candidate("Подходит", 8, relevance=0),
    ], required(), settings=Settings(_env_file=None))
    assert [item.candidate.name for item in results] == ["Подходит", "Неизвестно", "Недостаточно"]
    assert [item.score for item in results] == [100, 66.67, 0]
    assert [item.coverage for item in results] == [1, 0, 1]
    assert [item.rank for item in results] == [1, 2, 3]
    assert "доказательств недостаточно" in results[1].explanation


def test_coverage_counts_both_supported_outcomes_and_does_not_conflate_missing():
    requirements = [
        ParsedRequirement(name="Требование A", fact_type="a", type="number", operator="gte", expected_value=5),
        ParsedRequirement(name="Требование B", fact_type="b", type="number", operator="gte", expected_value=5),
        ParsedRequirement(name="Требование C", fact_type="c", type="number", operator="gte", expected_value=5),
    ]
    company = CompanyCandidate(name="Компания", facts=[
        FactData(fact_type="a", value=8, source_url="https://demo.example", source_quote="a=8"),
        FactData(fact_type="b", value=3, source_url="https://demo.example", source_quote="b=3"),
    ])
    result = rank_candidates([company], requirements, settings=Settings(_env_file=None))[0]
    assert (result.confirmed_count, result.contradicted_count, result.unknown_count) == (1, 1, 1)
    assert result.coverage == pytest.approx(2 / 3)
    assert result.score == pytest.approx(55.56)


def test_weights_are_configurable_and_formula_is_explained():
    config = Settings(_env_file=None, confirmed_weight=2, contradicted_weight=-4, unknown_weight=-1)
    result = rank_candidates([candidate("Компания")], required(), settings=config)[0]
    assert result.score == 50
    assert "сумма = -1" in result.explanation
    assert "Score =" in result.explanation


def test_no_requirements_is_neutral_and_relevance_only_breaks_ties():
    results = rank_candidates([candidate("Б", relevance=0.1), candidate("А", relevance=0.9)], [])
    assert [item.candidate.name for item in results] == ["А", "Б"]
    assert all(item.score == 0 and item.coverage == 0 and item.checks == [] for item in results)
    assert all("нет формализованных требований" in item.explanation for item in results)


def test_empty_candidates():
    assert rank_candidates([], required()) == []


def test_historical_wins_are_bounded_deduplicated_and_explained():
    winner = candidate("Победитель")
    win = FactData(fact_type="historical_win", value=True, source_kind="history",
                   source_url="history://records/lot/1", source_quote='"is_winner": true')
    winner.facts = [win, win.model_copy(), win.model_copy(update={"source_url": "history://records/lot/2"})]
    result = rank_candidates([winner], required(), settings=Settings(_env_file=None))[0]
    assert result.history_wins == 2
    assert result.history_bonus == 4
    assert result.base_score == 66.67
    assert result.score == 70.67
    assert result.unknown_count == 1
    assert "не подтверждает исполнение" in result.explanation
    winner.facts.extend(win.model_copy(update={"source_url": f"history://records/lot/{i}"}) for i in range(3, 20))
    assert rank_candidates([winner], required())[0].history_bonus == 5


def test_web_win_claim_and_false_do_not_create_history_bonus():
    company = candidate("Компания")
    company.facts = [
        FactData(fact_type="historical_win", value=True, source_url="https://company.example", source_quote="Мы выигрывали"),
        FactData(fact_type="historical_win", value=False, source_kind="history", source_url="history://lot/1", source_quote="false"),
    ]
    result = rank_candidates([company], required())[0]
    assert result.history_wins == result.history_bonus == 0
