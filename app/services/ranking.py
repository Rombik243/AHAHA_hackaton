"""Transparent configurable ranking, independent of LLM output and retrieval."""
from app.config import Settings, get_settings
from app.schemas.domain import CompanyCandidate, ParsedRequirement, RankedCandidate
from app.services.requirement_checker import check_requirements


def rank_candidates(
    candidates: list[CompanyCandidate],
    requirements: list[ParsedRequirement],
    *,
    settings: Settings | None = None,
) -> list[RankedCandidate]:
    """Min-max normalize the sum of configured requirement weights to [0, 100].

    score = 100 * (raw - N * contradicted_weight)
                    / (N * (confirmed_weight - contradicted_weight)).
    Thus default all-unknown score is 66.67, not a probability of suitability.
    Coverage is the share of checks with a supported conclusion. Empty criteria
    have neutral score/coverage zero, with retrieval relevance as a tie-breaker.
    """
    config = settings or get_settings()
    weights = {
        "confirmed": config.confirmed_weight,
        "contradicted": config.contradicted_weight,
        "unknown": config.unknown_weight,
    }
    results = []
    total = len(requirements)
    for candidate in candidates:
        checks = check_requirements(requirements, candidate.facts)
        confirmed = sum(check.status == "confirmed" for check in checks)
        contradicted = sum(check.status == "contradicted" for check in checks)
        unknown = total - confirmed - contradicted
        raw = sum(weights[check.status] for check in checks)
        if total:
            score = 100 * (raw - total * config.contradicted_weight) / (
                total * (config.confirmed_weight - config.contradicted_weight)
            )
            coverage = (confirmed + contradicted) / total
            explanation = (
                f"Подтверждено: {confirmed}; противоречит: {contradicted}; нет данных: {unknown}. "
                f"Покрытие проверок: {coverage:.0%}. "
                f"Веса: {config.confirmed_weight:g}, {config.contradicted_weight:g}, "
                f"{config.unknown_weight:g}; сумма = {raw:g}. "
                f"Score = 100 × (сумма − {total} × {config.contradicted_weight:g}) / "
                f"({total} × ({config.confirmed_weight:g} − ({config.contradicted_weight:g}))). "
                "Это нормированный индекс по требованиям, а не вероятность соответствия; "
                "отсутствие данных требует проверки."
            )
            if unknown == total:
                explanation += " Все требования имеют статус unknown: доказательств недостаточно."
        else:
            score, coverage = 0.0, 0.0
            explanation = (
                "В лоте нет формализованных требований: score и покрытие равны 0. "
                "Компании упорядочены только по поисковой релевантности."
            )
        base_score = max(0.0, min(100.0, score))
        wins = len({fact.source_url for fact in candidate.facts if (
            fact.fact_type == "historical_win" and fact.value is True
            and fact.source_kind == "history" and fact.source_url.startswith("history://")
        )})
        bonus = min(config.max_history_bonus, wins * config.history_win_bonus, 100.0 - base_score) if total else 0.0
        score = base_score + bonus
        if wins:
            explanation += (
                f" Побед в сопоставимой истории: {wins}; бонус к базовому индексу: {bonus:g}. "
                "Победа не подтверждает исполнение контракта и текущие требования."
            )
        results.append(RankedCandidate(
            candidate=candidate,
            checks=checks,
            score=round(max(0.0, min(100.0, score)), 2),
            base_score=round(base_score, 2), history_bonus=round(bonus, 2), history_wins=wins,
            confirmed_count=confirmed,
            contradicted_count=contradicted,
            unknown_count=unknown,
            coverage=coverage,
            explanation=explanation,
        ))
    results.sort(key=lambda item: (-item.score, -item.candidate.relevance, -item.history_wins, item.candidate.name.casefold()))
    for rank, item in enumerate(results, 1):
        item.rank = rank
    return results
