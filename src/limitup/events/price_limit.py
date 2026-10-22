"""01-2, 03-1 호가가격단위와 상한가·하한가 계산

호가가격단위는 2023-01-25에 개편됐다. 그 전에는 KOSPI와 KOSDAQ의 단위표가 달랐으므로
2021~2022년 데이터에는 이전 단위표를 써야 한다.
"""

import datetime as dt

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


def upper_limit(base, date=None, market="KOSPI"):
    """상한가 = 기준가격 x 1.3 을 그 가격대의 호가단위로 버림"""
    raw = base * 13 // 10
    tick = tick_size(raw, date, market)
    return raw // tick * tick


def lower_limit(base, date=None, market="KOSPI"):
    """하한가 = 기준가격 x 0.7 을 그 가격대의 호가단위로 올림"""
    raw = -(-base * 7 // 10)                   # 올림 정수 나눗셈
    tick = tick_size(raw, date, market)
    return -(-raw // tick) * tick
