import json
from typing import List, Optional


def load_id_to_country(country_map_json: Optional[str]) -> Optional[List[str]]:
    if not country_map_json:
        return None
    with open(country_map_json, "r", encoding="utf-8") as f:
        m = json.load(f)  # country(str)->id(int)
    inv = [""] * (max(m.values()) + 1)
    for k, v in m.items():
        inv[int(v)] = str(k)
    return inv
