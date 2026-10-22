"""07-1, 07-5 설정 파일을 읽어 매수 후보를 고른다

    from limitup.strategies.rules import load_config, select
    cfg = load_config()
    picks = select(events_with_scores, cfg, "B")
"""

import tomllib

import numpy as np
import pandas as pd

from limitup.paths import ROOT

CONFIG = ROOT / "configs" / "strategies.toml"


def load_config(path=CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def eligible(d: pd.DataFrame, common: dict) -> pd.Series:
    """매수 제외 조건 (07-5). True면 매수 후보가 될 수 있다."""
    ok = d.f_value_log >= np.log10(common["min_value_krw"])
    if common.get("exclude_risk_material") and "l_group" in d:
        ok &= d.l_group != "risk"
    return ok


def select(d: pd.DataFrame, cfg: dict, name: str) -> pd.DataFrame:
    """전략 name 의 규칙으로 날마다 상위 종목을 고르고, 유동성 한도로 매수 금액을 정한다."""
    common, rule = cfg["common"], cfg["strategy"][name]
    c = d[eligible(d, common)]
    if rule.get("skip_if_open_at_limit"):
        c = c[c.t1_open < c.t1_upper]
    picks = c.sort_values(rule["select_by"], ascending=False).groupby("date").head(common["max_positions_per_day"])
    value = 10 ** picks.f_value_log
    picks = picks.assign(position_krw=np.minimum(common["position_krw"], value * common["max_value_share"]))
    return picks.sort_values("date")
