from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


GOLDEN_COUNTRY_NAME_TO_CODE = {
    "Albania": "AL",
    "Andorra": "AD",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Bangladesh": "BD",
    "Belarus": "BY",
    "Belgium": "BE",
    "Bermuda": "BM",
    "Bhutan": "BT",
    "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA",
    "Botswana": "BW",
    "Brazil": "BR",
    "Bulgaria": "BG",
    "Cambodia": "KH",
    "Canada": "CA",
    "Chile": "CL",
    "Christmas Island": "CX",
    "Cocos (Keeling) Islands": "CC",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Ecuador": "EC",
    "Estonia": "EE",
    "Faroe Islands": "FO",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "Gibraltar": "GI",
    "Greece": "GR",
    "Greenland": "GL",
    "Hong Kong": "HK",
    "Hungary": "HU",
    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Ireland": "IE",
    "Isle of Man": "IM",
    "Israel": "IL",
    "Italy": "IT",
    "Japan": "JP",
    "Jersey": "JE",
    "Jordan": "JO",
    "Kazakhstan": "KZ",
    "Kenya": "KE",
    "Kyrgyzstan": "KG",
    "Laos": "LA",
    "Latvia": "LV",
    "Lebanon": "LB",
    "Lesotho": "LS",
    "Liechtenstein": "LI",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malaysia": "MY",
    "Malta": "MT",
    "Mexico": "MX",
    "Monaco": "MC",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Namibia": "NA",
    "Nepal": "NP",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nigeria": "NG",
    "North Macedonia": "MK",
    "Norway": "NO",
    "Oman": "OM",
    "Panama": "PA",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Romania": "RO",
    "Russia": "RU",
    "Rwanda": "RW",
    "R\u00e9union": "RE",
    "San Marino": "SM",
    "Senegal": "SN",
    "Serbia": "RS",
    "Singapore": "SG",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Taiwan": "TW",
    "Thailand": "TH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Uganda": "UG",
    "Ukraine": "UA",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States": "US",
    "Uruguay": "UY",
    "Vietnam": "VN",
    "\u00c5land": "AX",
}

COUNTRY_ALIAS_PREFIX_TO_CODE = {
    "United States": "US",
}

COUNTRY_CODE_TO_NAME: dict[str, str] = {}
for name, code in GOLDEN_COUNTRY_NAME_TO_CODE.items():
    COUNTRY_CODE_TO_NAME.setdefault(code, name)

COUNTRY_MAPPING_PATH = Path(__file__).with_name("country_mapping.json")


@lru_cache
def _load_country_mapping() -> dict:
    if not COUNTRY_MAPPING_PATH.exists():
        return {}
    try:
        with COUNTRY_MAPPING_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def golden_country_to_code(name: str) -> Optional[str]:
    if not name:
        return None
    name = str(name).strip()
    mapping = _load_country_mapping()
    alias_exact = mapping.get("alias_to_code") if isinstance(mapping.get("alias_to_code"), dict) else {}
    code_by_name = mapping.get("code_by_name") if isinstance(mapping.get("code_by_name"), dict) else GOLDEN_COUNTRY_NAME_TO_CODE
    alias_prefix = mapping.get("alias_prefix_to_code") if isinstance(mapping.get("alias_prefix_to_code"), dict) else COUNTRY_ALIAS_PREFIX_TO_CODE

    if name in alias_exact:
        return alias_exact.get(name)
    if name in code_by_name:
        return code_by_name.get(name)
    for prefix, code in alias_prefix.items():
        if name.startswith(prefix):
            return code
    return None


def country_code_to_name(code: str) -> Optional[str]:
    if not code:
        return None
    code = str(code).strip().upper()
    mapping = _load_country_mapping()
    canonical_by_code = mapping.get("canonical_by_code") if isinstance(mapping.get("canonical_by_code"), dict) else COUNTRY_CODE_TO_NAME
    return canonical_by_code.get(code, code)
