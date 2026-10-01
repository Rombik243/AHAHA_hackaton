"""Company identity storage; conflicting known INNs always remain separate."""
import re
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Company
from app.schemas.domain import CompanyCandidate


def normalized_name(name: str) -> str:
    words = re.findall(r"[\w]+", name.casefold().replace("ё", "е"))
    legal_forms = {"ооо", "оао", "зао", "пао", "ао", "ип", "llc", "ltd"}
    return " ".join(word for word in words if word not in legal_forms)


def normalized_domain(website: str | None) -> str | None:
    if not website:
        return None
    try:
        hostname = urlsplit(website if "://" in website else "https://" + website).hostname
        return hostname.casefold().removeprefix("www.") if hostname else None
    except ValueError:
        return None


async def upsert_company(
    session: AsyncSession, candidate: CompanyCandidate, *, commit: bool = True,
) -> Company:
    """Commit by default. The lot-results transaction calls with commit=False."""
    name = normalized_name(candidate.name)
    domain = normalized_domain(candidate.website)
    company = None
    if candidate.inn:
        company = await session.scalar(select(Company).where(Company.inn == candidate.inn))
    if company is None:
        conditions = []
        if domain:
            conditions.append(Company.domain == domain)
        if name:
            conditions.append(Company.normalized_name == name)
        for condition in conditions:
            matches = list((await session.scalars(select(Company).where(condition))).all())
            # Ambiguous shared domains/names never collapse several legal entities.
            if len(matches) != 1:
                continue
            match = matches[0]
            if candidate.inn and match.inn and candidate.inn != match.inn:
                continue
            if domain and match.domain and domain != match.domain:
                continue
            company = match
            break
    if company is None:
        company = Company(
            name=candidate.name, inn=candidate.inn, website=candidate.website,
            normalized_name=name, domain=domain,
        )
        session.add(company)
    else:
        if not company.inn and candidate.inn:
            company.inn = candidate.inn
        if not company.website and candidate.website:
            company.website = candidate.website
            company.domain = domain
    await session.flush()
    if commit:
        await session.commit()
    return company
