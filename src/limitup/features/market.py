"""04-3 시장 환경과 업종(테마 근사) 피처 — T일 장 마감 시점에 알 수 있는 시장 전체 정보

업종은 한국표준산업분류 기반의 현재 분류를 쓴다(02-3, collectors.master). 테마와 같지 않으므로
'같은 업종이 함께 급등했는가'를 테마 동반 상승의 근사치로 쓴다. LLM 기반 재료 분석은 06장.
"""

import pandas as pd

from limitup.paths import INDEX_FILE, INDUSTRY

MARKET_FEATURES = ["m_idx_ret", "m_idx_ret5", "m_n_lu", "m_lu_heat", "m_breadth",
                   "s_ind_strong", "s_ind_lu", "s_leader"]
STRONG = 1.10                       # 업종 동반 급등 기준: 기준가 대비 +10% 이상 마감


def market_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜별 시장 피처: 상한가 마감 종목 수, 그 20일 평균 대비 배수, 상승 종목 비율"""
    t = df[~df.halted & ~df.listing_day & ~df.liquidation]
    by = t.groupby("date").agg(m_n_lu=("lu_close", "sum"),
                               m_breadth=("close", "size"))
    by["m_breadth"] = t.assign(up=t.close > t.base).groupby("date").up.mean()
    by["m_lu_heat"] = by.m_n_lu / by.m_n_lu.shift(1).rolling(20).mean()
    return by.reset_index()


def index_features() -> pd.DataFrame:
    """시장(KOSPI·KOSDAQ)별 지수 당일 수익률과 5일 수익률"""
    ix = pd.read_parquet(INDEX_FILE).sort_values(["index", "date"])
    g = ix.groupby("index").close
    ix["m_idx_ret"] = ix.close / g.shift(1) - 1
    ix["m_idx_ret5"] = ix.close / g.shift(5) - 1
    return ix.rename(columns={"index": "market"})[["market", "date", "m_idx_ret", "m_idx_ret5"]]


def industry_features(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """같은 업종의 당일 급등·상한가 종목 수(자기 자신 제외)와 업종 내 거래대금 1위 여부"""
    ind = pd.read_parquet(INDUSTRY)[["code", "industry"]]
    days = df[df.date.isin(events.date.unique()) & ~df.halted][["code", "date", "close", "base", "value", "lu_close"]]
    days = days.merge(ind, on="code", how="left")
    days["strong"] = days.close >= days.base * STRONG
    g = days.groupby(["date", "industry"])
    days["n_strong"] = g.strong.transform("sum")
    days["n_lu"] = g.lu_close.transform("sum")
    top = days[days.strong].sort_values("value").groupby(["date", "industry"]).code.last()
    days["s_leader"] = [int(top.get((d, i)) == c) for c, d, i in zip(days.code, days.date, days.industry)]
    days["s_ind_strong"] = days.n_strong - days.strong.astype(int)     # 자기 자신 제외
    days["s_ind_lu"] = days.n_lu - days.lu_close.astype(int)
    return events.merge(days[["code", "date", "industry", "s_ind_strong", "s_ind_lu", "s_leader"]],
                        on=["code", "date"], how="left")


def market_features(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = events.merge(market_daily_features(df), on="date", how="left")
    out = out.merge(index_features(), on=["market", "date"], how="left")
    return industry_features(out, df)
