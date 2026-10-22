"""04-2 장중 피처 — T일 1분봉으로 상한가가 어떻게 만들어졌는지 수치화한다

분봉은 최근 약 1년만 조회되므로(2025-10-14~), 이 피처는 그 기간의 이벤트에만 있다.
"""

import numpy as np
import pandas as pd

from limitup.collectors.minute import MINUTE_DIR, load_minute

INTRADAY_FEATURES = ["i_first_min", "i_breaks", "i_hold", "i_vol_after", "i_close_auction"]
TYPES = {"early": "조기", "gradual": "점진", "repeat": "반복 이탈", "late": "막판"}


def minutes_since_open(times: pd.Series) -> pd.Series:
    """'HHMMSS' 문자열을 그날 첫 분봉으로부터 지난 분으로 바꾼다 (수능일처럼 늦게 열어도 맞게)"""
    t = pd.to_datetime(times, format="%H%M%S")
    return (t - t.min()).dt.total_seconds() / 60


def intraday_one(m: pd.DataFrame, upper: int) -> dict:
    """하루 분봉과 그날 상한가로 장중 피처를 계산한다."""
    m = m.sort_values("time").reset_index(drop=True)
    mins = minutes_since_open(m.time)
    hit = m.high >= upper
    if not hit.any():
        return {}
    i0 = int(hit.idxmax())                                     # 처음 상한가에 닿은 분봉
    after = m.iloc[i0:]
    at_limit = after.close >= upper
    breaks = int((~at_limit & at_limit.shift(1, fill_value=True)).sum())   # 상한가 → 이탈 전환 횟수
    total = m.volume.sum()
    return {
        "i_first_min": float(mins.iloc[i0]),
        "i_breaks": breaks,
        "i_hold": float(at_limit.mean()),                      # 첫 도달 이후 상한가로 끝난 분봉 비율
        "i_vol_after": after.volume.sum() / total if total else np.nan,
        "i_close_auction": m.volume.iloc[-1] / total if total else np.nan,
    }


def limit_type(row) -> str:
    """상한가 형성 유형 (01-3의 A~D)"""
    if row.i_breaks >= 3:
        return "repeat"
    if row.i_first_min < 60 and row.i_breaks <= 1:
        return "early"
    if row.i_first_min >= 330:                                 # 장 시작 5시간 30분 뒤 (보통 14:30) 이후
        return "late"
    return "gradual"


def intraday_features(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in events.itertuples():
        day = r.date.strftime("%Y%m%d")
        if not (MINUTE_DIR / day / f"{r.code}.parquet").exists():
            continue
        m = load_minute(r.code, day)
        if m.empty:
            continue
        f = intraday_one(m, r.upper)
        if f:
            rows.append({"code": r.code, "date": r.date, **f})
    feat = pd.DataFrame(rows, columns=["code", "date", *INTRADAY_FEATURES])
    feat["i_type"] = feat.apply(limit_type, axis=1) if len(feat) else []
    return events.merge(feat, on=["code", "date"], how="left")
