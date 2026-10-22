"""01-2, 03-1 호가가격단위와 상한가·하한가 계산

호가가격단위는 2023-01-25에 개편됐다. 그 전에는 KOSPI와 KOSDAQ의 단위표가 달랐으므로
2021~2022년 데이터에는 이전 단위표를 써야 한다.
"""

import datetime as dt

import numpy as np
import pandas as pd

TICK_REFORM = dt.date(2023, 1, 25)

# (구간 상한, 호가단위) — 마지막 구간은 상한 없음
TICKS_NEW = [(2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
             (200_000, 100), (500_000, 500), (None, 1_000)]
TICKS_OLD_KOSPI = [(1_000, 1), (5_000, 5), (10_000, 10), (50_000, 50),
                   (100_000, 100), (500_000, 500), (None, 1_000)]
TICKS_OLD_KOSDAQ = [(1_000, 1), (5_000, 5), (10_000, 10), (50_000, 50), (None, 100)]


def tick_table(date=None, market="KOSPI"):
    if date is None or date >= TICK_REFORM:
        return TICKS_NEW
    return TICKS_OLD_KOSDAQ if market == "KOSDAQ" else TICKS_OLD_KOSPI


def tick_size(price, date=None, market="KOSPI"):
    for bound, tick in tick_table(date, market):
        if bound is None or price < bound:
            return tick


def limit_width(base, date=None, market="KOSPI"):
    """가격제한폭 = 기준가격 x 30% 를 '기준가격'의 호가단위로 버린 금액 (03-1 실측으로 확인)"""
    tick = tick_size(base, date, market)
    return base * 3 // 10 // tick * tick


def upper_limit(base, date=None, market="KOSPI"):
    """상한가 = 기준가격 + 가격제한폭 을 그 가격대의 호가단위로 버림"""
    raw = base + limit_width(base, date, market)
    tick = tick_size(raw, date, market)
    return raw // tick * tick


def lower_limit(base, date=None, market="KOSPI"):
    """하한가 = 기준가격 - 가격제한폭 을 그 가격대의 호가단위로 올림"""
    raw = base - limit_width(base, date, market)
    tick = tick_size(raw, date, market)
    return -(-raw // tick) * tick


# ---- 03-1 전 종목에 한 번에 적용하는 벡터 버전 ----

def _tick_vec(price, date, market):
    """가격·날짜·시장 배열에 맞는 호가단위 배열"""
    def pick(table):
        bounds = [b for b, _ in table[:-1]]
        ticks = [t for _, t in table]
        return np.select([price < b for b in bounds], ticks[:-1], ticks[-1])
    new = date >= np.datetime64(TICK_REFORM)
    kosdaq = market == "KOSDAQ"
    return np.where(new, pick(TICKS_NEW),
                    np.where(kosdaq, pick(TICKS_OLD_KOSDAQ), pick(TICKS_OLD_KOSPI)))


def add_limits(df: pd.DataFrame) -> pd.DataFrame:
    """일봉(date, market, base, high, close)에 상한가·하한가와 상한가 터치·마감 여부를 더한다."""
    base = df.base.to_numpy("int64")
    date = df.date.to_numpy("datetime64[ns]")
    market = df.market.to_numpy()
    base_tick = _tick_vec(base, date, market)
    width = base * 3 // 10 // base_tick * base_tick          # 가격제한폭
    up_raw, lo_raw = base + width, base - width
    up_tick = _tick_vec(up_raw, date, market)
    lo_tick = _tick_vec(lo_raw, date, market)
    out = df.copy()
    out["upper"] = up_raw // up_tick * up_tick
    out["lower"] = -(-lo_raw // lo_tick) * lo_tick
    traded = out.volume > 0
    out["lu_touch"] = traded & (out.high == out.upper)     # 장중 상한가 터치
    out["lu_close"] = traded & (out.close == out.upper)    # 상한가 마감
    return out
