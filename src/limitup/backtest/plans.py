"""08-2, 08-3 전략별 주문 계획과 체결 모형

전략 B-마감 : T일 장 마감 단일가에 상한가 매수 → T+1 시가 매도 (07장 설계, 점수로 하루 1종목)
  - optimistic : 항상 체결 (07장의 가정)
  - strict     : 상한가로 끝난 종목의 장 마감 단일가 매수는 배분 여부를 데이터로 확인할 수 없음 → 미체결
전략 B-터치 : 첫 상한가에 처음 닿는 순간 상한가 매수 → T+1 시가 매도 (분봉이 있는 최근 1년)
  - queue      : 주문 뒤 상한가 아래 체결이 나오면 대기열이 소진된 것이므로 체결, 끝까지 잠기면 미체결
전략 C      : T+1 시가 매수 → 익절·손절(일봉, 손절 우선) 또는 T+1 종가 매도
"""

import numpy as np
import pandas as pd

from limitup.backtest.engine import Order
from limitup.collectors.minute import MINUTE_DIR, load_minute
from limitup.strategies.design import exit_with_stops

T_CLOSE, T1_OPEN, T1_CLOSE = pd.Timedelta(hours=15, minutes=30), pd.Timedelta(hours=9), pd.Timedelta(hours=15, minutes=30)


def b_close_orders(picks: pd.DataFrame, fill: str, slip_sell=0.0) -> list:
    out = []
    for r in picks.itertuples():
        exit_px = r.t1_open * (r.upper / r.t1_base)            # T+1 기준가가 바뀌는 수정 사건 보정
        out.append(Order(r.code, r.date + T_CLOSE, r.upper, r.t1_date + T1_OPEN, exit_px * (1 - slip_sell),
                         r.position_krw, filled=(fill == "optimistic"), tag="B-close"))
    return out


def first_touch(m: pd.DataFrame, upper: int):
    """처음 상한가에 닿은 분봉 번호와 시각"""
    m = m.sort_values("time").reset_index(drop=True)
    hit = np.flatnonzero(m.high.to_numpy() >= upper)
    return (m, int(hit[0])) if len(hit) else (m, None)


def queue_filled(m: pd.DataFrame, i0: int, upper: int, delay: int = 0) -> bool:
    """주문(첫 도달 분봉 + delay분)보다 뒤 분봉에서 상한가 아래 체결이 있었는가 = 대기열 소진"""
    after = m.iloc[i0 + delay + 1:]
    return bool((after.low < upper).any())


def b_touch_candidates(events: pd.DataFrame, delay: int = 0) -> pd.DataFrame:
    """분봉이 있는 첫 상한가(마감·터치 집단)마다 첫 도달 시각과 대기열 체결 여부를 계산한다."""
    rows = []
    for r in events.itertuples():
        day = f"{r.date:%Y%m%d}"
        if not (MINUTE_DIR / day / f"{r.code}.parquet").exists():
            continue
        m = load_minute(r.code, day)
        if m.empty:
            continue
        m, i0 = first_touch(m, r.upper)
        if i0 is None:
            continue
        rows.append({"code": r.code, "date": r.date, "group": r.group, "touch_time": m.time.iloc[i0],
                     "filled": queue_filled(m, i0, r.upper, delay)})
    return events.merge(pd.DataFrame(rows), on=["code", "date", "group"])


def b_touch_orders(cand: pd.DataFrame, amount: float, slip_sell=0.0) -> list:
    """하루에 가장 먼저 상한가에 닿은 한 종목만 (그 시점에 알 수 있는 선착순 규칙)"""
    first = cand.sort_values("touch_time").groupby("date").head(1)
    out = []
    for r in first.itertuples():
        t = r.date + pd.to_timedelta(f"{r.touch_time[:2]}:{r.touch_time[2:4]}:00")
        exit_px = r.t1_open * (r.close / r.t1_base)
        out.append(Order(r.code, t, r.upper, r.t1_date + T1_OPEN, exit_px * (1 - slip_sell), amount,
                         filled=r.filled, tag=f"B-touch:{r.group}"))
    return out


def c_orders(picks: pd.DataFrame, tp: float, sl: float, slip=0.0) -> list:
    out = []
    for r in picks.itertuples():
        entry = r.t1_open * (1 + slip)
        ret = exit_with_stops(np.array([r.t1_open]), np.array([r.t1_high]), np.array([r.t1_low]),
                              np.array([r.t1_close]), np.array([r.t1_upper]), tp, sl)[0]
        exit_px = r.t1_open * (1 + ret) * (1 - slip)
        out.append(Order(r.code, r.t1_date + T1_OPEN, entry, r.t1_date + T1_CLOSE, exit_px, r.position_krw,
                         tag="C"))
    return out
