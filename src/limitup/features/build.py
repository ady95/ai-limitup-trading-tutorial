"""04-4, 04-5 피처·라벨 데이터셋 조립, 기간 분할, 미래정보 누수 검사

    uv run python -m limitup.features.build             # data/processed/features.parquet
    uv run python -m limitup.features.build --leak 300  # 누수 검사 (무작위 이벤트 300건)
    uv run python -m limitup.features.build --explore   # 학습 기간 안에서 피처별 구간 통계

의사결정 시각: T일 장 마감 후(15:30 이후) ~ T+1일 장 시작 전. 피처는 T일 종가까지의 정보만 쓴다.
"""

import argparse

import numpy as np
import pandas as pd

from limitup.events.detect import load_daily
from limitup.features.daily import DAILY_FEATURES, daily_features, stock_features
from limitup.features.intraday import INTRADAY_FEATURES, intraday_features
from limitup.features.market import (MARKET_FEATURES, industry_features, market_daily_features,
                                     market_features)
from limitup.paths import PROCESSED

FEATURES_FILE = PROCESSED / "features.parquet"
LABELS = {"t1_lu_touch": "y_touch", "t1_lu_close": "y_close", "p3": "y_p3",
          "t1_open_ret": "y_ret_open", "t1_close_ret": "y_ret_close", "t1_oc_ret": "y_ret_oc",
          "t1_low_ret": "y_ret_low"}
# 04-5 기간 분할: 일봉 피처는 연도별, 장중 피처는 분봉이 있는 최근 1년 안에서 시간순
SPLITS = [("train", "2021-01-01"), ("valid", "2024-01-01"), ("test", "2025-01-01")]
INTRADAY_START, INTRADAY_SPLIT = "2025-10-14", "2026-06-01"
NXT_START = "2025-03-04"


def split_of(dates: pd.Series) -> pd.Series:
    out = pd.Series("train", index=dates.index)
    for name, start in SPLITS:
        out[dates >= start] = name
    return out


def build(df=None) -> pd.DataFrame:
    df = load_daily() if df is None else df
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[(ev.group == "close") & ev.has_t1].copy()
    ds = ev[["code", "date", "market", "upper", "t3_date", *LABELS]].rename(columns=LABELS)
    for c in ("y_touch", "y_close", "y_p3"):
        ds[c] = ds[c].astype(int)
    ds = daily_features(ds, df)
    ds = intraday_features(ds)
    ds = market_features(ds, df)
    ds["split"] = split_of(ds.date)
    # 퍼지(purge): 라벨이 보는 기간(최대 T+3일)이 다음 분할 구간으로 넘어가는 이벤트
    t3 = ds.t3_date.fillna(ds.date)
    ds["purge"] = split_of(t3) != ds.split
    ds["after_nxt"] = (ds.date >= NXT_START).astype(int)
    ds["has_intraday"] = ds.i_first_min.notna()
    ds["i_split"] = np.where(~ds.has_intraday, None,
                             np.where(ds.date < INTRADAY_SPLIT, "train", "test"))
    ds["i_purge"] = ds.has_intraday & (ds.date < INTRADAY_SPLIT) & (t3 >= INTRADAY_SPLIT)
    return ds.sort_values(["date", "code"], ignore_index=True)


def leak_check(ds: pd.DataFrame, df: pd.DataFrame, n=300, seed=0) -> pd.DataFrame:
    """미래 데이터를 지운 뒤 다시 계산해도 피처가 같은지 확인한다.

    무작위로 고른 이벤트마다 T일 이후의 일봉을 모두 지우고 일봉·시장·업종 피처를 다시 계산해 비교한다.
    """
    rows = []
    for r in ds.sample(n, random_state=seed).itertuples():
        past = df[df.date <= r.date]
        f = stock_features(past[past.code == r.code])
        f = f[f.date == r.date].iloc[0]
        m = market_daily_features(past).set_index("date").loc[r.date]
        one = pd.DataFrame({"code": [r.code], "date": [r.date]})
        ind = industry_features(one, past).iloc[0]
        for col in DAILY_FEATURES:
            rows.append((col, getattr(r, col), f[col]))
        for col in ("m_n_lu", "m_lu_heat", "m_breadth"):
            rows.append((col, getattr(r, col), m[col]))
        for col in ("s_ind_strong", "s_ind_lu", "s_leader"):
            rows.append((col, getattr(r, col), ind[col]))
    t = pd.DataFrame(rows, columns=["feature", "full", "past_only"])
    same = np.isclose(t.full.astype(float), t.past_only.astype(float), equal_nan=True, rtol=1e-9)
    return t.assign(same=same).groupby("feature").same.agg(["size", "mean"])


def explore(ds: pd.DataFrame):
    """학습 기간(train) 안에서만 피처를 구간으로 나눠 T+1 상한가 마감 비율을 본다."""
    tr = ds[ds.split == "train"]
    print(f"학습 기간 이벤트 {len(tr):,}건, T+1 상한가 마감 {tr.y_close.mean():.1%}")
    for col in DAILY_FEATURES + ["m_idx_ret", "m_n_lu", "m_lu_heat", "m_breadth", "s_ind_strong"]:
        if tr[col].nunique() <= 2:
            g = tr.groupby(col)
        else:
            g = tr.groupby(pd.qcut(tr[col], 5, duplicates="drop"), observed=True)
        t = g.agg(n=("y_close", "size"), y_close=("y_close", "mean"), ret_oc=("y_ret_oc", "mean"))
        print(f"\n[{col}]\n" + t.to_string(formatters={"y_close": "{:.1%}".format, "ret_oc": "{:.1%}".format}))
    it = ds[ds.i_split == "train"]
    print(f"\n장중 피처 학습 구간 {len(it):,}건 (분봉 있는 {INTRADAY_START}~{INTRADAY_SPLIT} 이전)")
    for col in ["i_first_min", "i_breaks", "i_hold", "i_vol_after", "i_close_auction", "i_type"]:
        g = it.groupby(col) if col in ("i_type",) or it[col].nunique() <= 4 else \
            it.groupby(pd.qcut(it[col], 4, duplicates="drop"), observed=True)
        t = g.agg(n=("y_close", "size"), y_close=("y_close", "mean"), ret_oc=("y_ret_oc", "mean"))
        print(f"\n[{col}]\n" + t.to_string(formatters={"y_close": "{:.1%}".format, "ret_oc": "{:.1%}".format}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leak", type=int, default=0, help="누수 검사할 무작위 이벤트 수")
    ap.add_argument("--explore", action="store_true")
    args = ap.parse_args()

    df = load_daily()
    if FEATURES_FILE.exists() and (args.leak or args.explore):
        ds = pd.read_parquet(FEATURES_FILE)
    else:
        ds = build(df)
        ds.to_parquet(FEATURES_FILE, index=False)
        feats = DAILY_FEATURES + INTRADAY_FEATURES + MARKET_FEATURES
        print(f"데이터셋 {len(ds):,}건 x 피처 {len(feats)}개 → {FEATURES_FILE}")
        print("결측 비율 (0이 아닌 것):")
        na = ds[feats].isna().mean()
        print(na[na > 0].map("{:.1%}".format).to_string())
        print("\n기간 분할:")
        print(ds.groupby("split").agg(n=("y_close", "size"), first=("date", "min"), last=("date", "max"),
                                      y_close=("y_close", "mean")).to_string())
        print("\n장중 피처 분할:")
        print(ds[ds.has_intraday].groupby("i_split").agg(n=("y_close", "size"), first=("date", "min"),
                                                         last=("date", "max"), y_close=("y_close", "mean")).to_string())
    if args.leak:
        print(leak_check(ds, df, args.leak).to_string())
    if args.explore:
        explore(ds)


if __name__ == "__main__":
    main()
