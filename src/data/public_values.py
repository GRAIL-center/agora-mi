from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from data.io import normalize_tag
from runtime import load_yaml


@dataclass(frozen=True)
class PublicValueFamily:
    key: str
    label: str
    status: str
    description: str
    main_proxies: list[str]
    secondary_proxies: list[str]
    excluded_proxies: list[str]


@dataclass(frozen=True)
class GovernanceStrategyCategory:
    key: str
    label: str
    strategies: list[str]


@dataclass(frozen=True)
class PublicValueSpec:
    quality_filters: dict[str, Any]
    selection: dict[str, Any]
    families: dict[str, PublicValueFamily]
    governance_strategy_categories: dict[str, GovernanceStrategyCategory]


def load_public_value_spec(path: str | Path) -> PublicValueSpec:
    cfg = load_yaml(path)
    families: dict[str, PublicValueFamily] = {}
    for key, family_cfg in cfg.get("families", {}).items():
        families[key] = PublicValueFamily(
            key=key,
            label=str(family_cfg.get("label", key)),
            status=str(family_cfg.get("status", "main")),
            description=str(family_cfg.get("description", "")),
            main_proxies=list(family_cfg.get("main_proxies", [])),
            secondary_proxies=list(family_cfg.get("secondary_proxies", [])),
            excluded_proxies=list(family_cfg.get("excluded_proxies", [])),
        )

    strategy_categories: dict[str, GovernanceStrategyCategory] = {}
    for key, cat_cfg in cfg.get("governance_strategy_categories", {}).items():
        strategy_categories[key] = GovernanceStrategyCategory(
            key=key,
            label=str(cat_cfg.get("label", key)),
            strategies=list(cat_cfg.get("strategies", [])),
        )

    return PublicValueSpec(
        quality_filters=dict(cfg.get("quality_filters", {})),
        selection=dict(cfg.get("selection", {})),
        families=families,
        governance_strategy_categories=strategy_categories,
    )


def _normalized_members(values: Iterable[str]) -> set[str]:
    return {normalize_tag(value) for value in values if normalize_tag(value)}


def proxy_slug(proxy_name: str) -> str:
    return normalize_tag(proxy_name)


def family_proxies(
    family: PublicValueFamily,
    *,
    include_secondary: bool = True,
) -> list[tuple[str, str]]:
    pairs = [(proxy, "main") for proxy in family.main_proxies]
    if include_secondary:
        pairs.extend((proxy, "secondary") for proxy in family.secondary_proxies)
    return pairs


def all_family_proxy_rows(
    spec: PublicValueSpec,
    *,
    family_keys: Iterable[str] | None = None,
    include_secondary: bool = True,
    statuses: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    allowed_families = set(family_keys) if family_keys is not None else None
    allowed_statuses = set(statuses) if statuses is not None else None
    rows: list[dict[str, str]] = []
    for family_key, family in spec.families.items():
        if allowed_families is not None and family_key not in allowed_families:
            continue
        if allowed_statuses is not None and family.status not in allowed_statuses:
            continue
        for proxy_name, tier in family_proxies(family, include_secondary=include_secondary):
            rows.append(
                {
                    "family_key": family_key,
                    "family_label": family.label,
                    "family_status": family.status,
                    "proxy_name": proxy_name,
                    "proxy_slug": proxy_slug(proxy_name),
                    "proxy_tier": tier,
                }
            )
    return rows


def family_proxy_hits(tag_names: Iterable[str], family: PublicValueFamily) -> dict[str, Any]:
    tags_norm = _normalized_members(tag_names)
    main_hits = [proxy for proxy in family.main_proxies if normalize_tag(proxy) in tags_norm]
    secondary_hits = [proxy for proxy in family.secondary_proxies if normalize_tag(proxy) in tags_norm]
    excluded_hits = [proxy for proxy in family.excluded_proxies if normalize_tag(proxy) in tags_norm]
    has_main = bool(main_hits)
    has_any = has_main or bool(secondary_hits)
    if has_main and secondary_hits:
        tier = "mixed"
    elif has_main:
        tier = "main"
    elif secondary_hits:
        tier = "secondary"
    else:
        tier = "none"
    return {
        "matched": has_any,
        "tier": tier,
        "main_hits": main_hits,
        "secondary_hits": secondary_hits,
        "excluded_hits": excluded_hits,
    }


def assign_public_value_families(
    tag_names: Iterable[str],
    spec: PublicValueSpec,
    *,
    include_secondary: bool = True,
    require_main: bool = False,
    statuses: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    allowed_statuses = set(statuses) if statuses is not None else None
    hits: dict[str, dict[str, Any]] = {}
    for key, family in spec.families.items():
        if allowed_statuses is not None and family.status not in allowed_statuses:
            continue
        info = family_proxy_hits(tag_names, family)
        if not info["matched"]:
            continue
        if require_main and not info["main_hits"]:
            continue
        if not include_secondary and info["tier"] == "secondary":
            continue
        hits[key] = info
    return hits


def assign_governance_strategy_categories(
    tag_names: Iterable[str],
    spec: PublicValueSpec,
) -> dict[str, list[str]]:
    tags_norm = _normalized_members(tag_names)
    hits: dict[str, list[str]] = {}
    for key, category in spec.governance_strategy_categories.items():
        matched = [strategy for strategy in category.strategies if normalize_tag(strategy) in tags_norm]
        if matched:
            hits[key] = matched
    return hits
