"""06-3, 06-5 LLM·VLM 결과를 모델 입력 피처로 바꾸기

범주형 결과는 원-핫(0/1) 또는 순서 점수로 바꾼다. 공시가 없는 이벤트도 '공시 없음' 자체가 정보이므로
결측이 아니라 l_has_disc = 0 으로 둔다.
"""

import numpy as np
import pandas as pd

from limitup.llm.material import load_results as load_llm
from limitup.llm.vision import load_results as load_vlm

TYPE_GROUPS = {   # 표본이 적은 유형은 묶는다
    "supply_contract": "contract", "earnings": "earnings", "capital_raise": "financing",
    "control_change": "control", "mna": "control", "tech_pharma": "tech", "policy_theme": "theme",
    "legal_risk": "risk", "trading_alert": "risk", "shareholder_return": "other",
    "ownership_filing": "none", "routine": "none", "routine_only": "none", "other": "other",
}
LLM_FEATURES = ["l_has_disc", "l_n_disc", "l_n_material", "l_impact", "l_duration", "l_uncertainty",
                "l_confidence"] + [f"l_type_{g}" for g in sorted(set(TYPE_GROUPS.values()))]
VLM_FEATURES = ["v_early", "v_gradual", "v_repeated", "v_late", "v_lock", "v_sell", "v_front"]

IMPACT = {"negative": -1, "none": 0, "minor": 1, "moderate": 2, "significant": 3}
DURATION = {"one_day": 0, "short_term": 1, "medium_term": 2, "long_term": 3}
UNCERTAINTY = {"low": 0, "medium": 1, "high": 2}


def llm_features(ds: pd.DataFrame) -> pd.DataFrame:
    r = load_llm()
    f = pd.DataFrame({"code": r.code, "date": r.date})
    f["l_has_disc"] = r.has_disc.fillna(0).astype(int)
    f["l_n_disc"] = r.get("n_disc", pd.Series(0, index=r.index)).fillna(0)
    f["l_n_material"] = r.get("n_material", pd.Series(0, index=r.index)).fillna(0)
    f["l_impact"] = r.get("financial_impact").map(IMPACT)
    f["l_duration"] = r.get("impact_duration").map(DURATION)
    f["l_uncertainty"] = r.get("uncertainty").map(UNCERTAINTY)
    f["l_confidence"] = r.get("confidence")
    group = r.get("event_type").map(TYPE_GROUPS)
    group = group.where(f.l_has_disc == 1, "nodisc")
    for g in sorted(set(TYPE_GROUPS.values())):
        f[f"l_type_{g}"] = (group == g).astype(int)
    f["l_group"] = group.fillna("nodisc")
    return ds.merge(f, on=["code", "date"], how="left")


def vlm_features(ds: pd.DataFrame) -> pd.DataFrame:
    v = load_vlm()
    if v.empty:
        return ds.assign(**{c: np.nan for c in VLM_FEATURES})
    f = pd.DataFrame({"code": v.code, "date": v.date})
    for p, c in [("early_lock", "v_early"), ("gradual", "v_gradual"), ("repeated_break", "v_repeated"),
                 ("late", "v_late")]:
        f[c] = (v.pattern == p).astype(int)
    f["v_lock"] = v.lock_strength
    f["v_sell"] = v.selling_pressure
    f["v_front"] = v.volume_profile.map({"front_loaded": 1, "even": 0, "back_loaded": -1})
    f["v_pattern"] = v.pattern
    return ds.merge(f, on=["code", "date"], how="left")
