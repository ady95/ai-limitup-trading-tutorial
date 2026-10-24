"""03-2 첫 상한가 이벤트 탐지

    uv run python -m limitup.events.detect                 # 직전 60거래일 기준
    uv run python -m limitup.events.detect --lookback 20 40 60 120   # 민감도 비교

첫 상한가(T일) = 직전 lookback 거래일 동안 상한가 마감이 없던 종목의 상한가 마감.
결과는 data/processed/events.parquet 에 저장한다 (group: close = 마감 집단, touch = 터치 집단).
"""

import argparse
import os

import numpy as np
import pandas as pd

from limitup.events.price_limit import add_limits
from limitup.paths import PROCESSED
from limitup.storage import build

START = "2021-01-01"      # 이벤트 탐지 시작일 (그 전 데이터는 직전 거래일 확인용)
# 분석 기준일: 책의 수치는 2026-10-21(당시 확정된 마지막 거래일)로 고정해 재현할 수 있게 한다.
# 운영할 때는 LIMITUP_END 환경변수(또는 pipeline --latest)로 최신 확정일을 넣는다 (10-1).
END = os.environ.get("LIMITUP_END", "2026-10-21")
EVENTS = PROCESSED / "events.parquet"


def load_daily(end=END):
    """분석용 일봉: 당시 시장 기준(market_daily), 상한가·하한가와 제외 표시를 붙인다."""
    con = build()
    df = con.sql(f"""SELECT d.*, u.listed, u.delisted FROM market_daily d JOIN universe u USING (code)
                     WHERE date <= DATE '{end}' ORDER BY code, date""").df()
    df = add_limits(df)
    g = df.groupby("code", sort=False)
    df["first_row"] = g.cumcount() == 0
    df["listing_day"] = df.date == df.listed                       # 신규·이전 상장 첫날
    # 정리매매: 상장일이 아닌데 가격제한폭을 벗어난 날이 있으면, 그 6거래일 전부터 데이터 끝까지
    viol = (df.volume > 0) & ~df.listing_day & ((df.high > df.upper) | (df.low < df.lower))
    pos = g.cumcount()
    first_viol = pos.where(viol).groupby(df.code).transform("min")
    df["liquidation"] = first_viol.notna() & (pos >= first_viol - 6)
    df["halted"] = df.volume == 0
    df["prev_halted"] = g.halted.shift(1, fill_value=False)
    return df


def market_calendar(df):
    """시장 거래일: 거래된 종목이 500개 이상인 날"""
    n = df[~df.halted].groupby("date").size()
    return pd.DatetimeIndex(n[n >= 500].index).sort_values()


def detect(df, lookback=60, start=START):
    """첫 상한가 마감(close)과 첫 상한가 터치·이탈(touch) 이벤트를 찾는다."""
    cal = market_calendar(df)
    idx = pd.Series(np.arange(len(cal)), index=cal)
    d = df[df.date.isin(cal)].copy()
    d["ci"] = d.date.map(idx)                                      # 시장 거래일 번호

    # 직전 lookback 거래일 안의 상한가 마감 횟수 (자기 자신 제외)
    rows = []
    for code, s in d.groupby("code", sort=False):
        lu_ci = s.ci[s.lu_close & ~s.liquidation].to_numpy()
        ci = s.ci.to_numpy()
        left = np.searchsorted(lu_ci, ci - lookback, side="left")
        right = np.searchsorted(lu_ci, ci, side="left")
        rows.append(pd.Series(right - left, index=s.index))
    d["prior_lu"] = pd.concat(rows)
    d["history_ok"] = d.groupby("code").ci.transform("min") <= d.ci - lookback

    base_ok = (d.date >= start) & d.history_ok & (d.prior_lu == 0) \
        & ~d.listing_day & ~d.liquidation & ~d.prev_halted & ~d.halted
    ev = d[base_ok & (d.lu_close | d.lu_touch)].copy()
    ev["group"] = np.where(ev.lu_close, "close", "touch")
    return ev, d


def attach_outcomes(ev, d):
    """T+1 ~ T+3 결과를 붙인다. 수익률은 T+1 기준가(= T일 종가, 수정 사건 반영) 대비."""
    nxt = d.sort_values(["code", "date"]).groupby("code")
    cols = ["date", "open", "high", "low", "close", "base", "upper", "lu_touch", "lu_close",
            "halted", "liquidation"]
    out = ev.copy()
    for k in (1, 2, 3):
        sh = nxt[cols].shift(-k).loc[ev.index]
        sh.columns = [f"t{k}_{c}" for c in cols]
        out = out.join(sh)
    b = out.t1_base
    for c in ("open", "high", "low", "close"):
        out[f"t1_{c}_ret"] = out[f"t1_{c}"] / b - 1
    out["t1_oc_ret"] = out.t1_close / out.t1_open - 1          # 시가 매수 → 종가 매도
    out["t1_mfe"] = out.t1_high / out.t1_open - 1              # 시가 매수 후 최대 유리 변동폭
    out["t1_mae"] = out.t1_low / out.t1_open - 1               # 시가 매수 후 최대 불리 변동폭
    out["has_t1"] = out.t1_date.notna() & ~out.t1_halted.fillna(True).astype(bool)
    out["p3"] = out[["t1_lu_close", "t2_lu_close", "t3_lu_close"]].fillna(False).any(axis=1)
    keep = ["code", "market", "date", "group", "open", "high", "low", "close", "base", "upper",
            "volume", "value"] + [c for c in out.columns if c.startswith("t") and c[1] in "123"] \
        + ["has_t1", "p3"]
    return out[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, nargs="+", default=[60])
    args = ap.parse_args()

    df = load_daily()
    print(f"분석 일봉 {len(df):,}행 (상장 첫날 {df.listing_day.sum():,}, 정리매매 {df.liquidation.sum():,}, "
          f"거래정지 {df.halted.sum():,})")
    for lb in args.lookback:
        ev, d = detect(df, lb)
        ev = attach_outcomes(ev, d)
        c = ev[(ev.group == "close") & ev.has_t1]
        print(f"lookback {lb:>3}: 마감 {(ev.group == 'close').sum():>5,}  터치 {(ev.group == 'touch').sum():>5,}  "
              f"| 마감 집단 T+1 상한가 도달 {c.t1_lu_touch.mean():.1%}  마감 {c.t1_lu_close.mean():.1%}")
        if lb == 60:
            PROCESSED.mkdir(parents=True, exist_ok=True)
            ev.to_parquet(EVENTS, index=False)
            print(f"저장 → {EVENTS}")


if __name__ == "__main__":
    main()
