"""08-4 매매 비용 — 수수료, 증권거래세(매도 시), 슬리피지

증권거래세율(매도 금액 기준, KOSPI는 농어촌특별세 포함, KOSDAQ 동일):
  2021~2022 0.23%, 2023 0.20%, 2024 0.18%, 2025 0.15%, 2026 0.20%
수수료는 증권사·매체마다 다르다. 이 책은 온라인 거래 수준인 편도 0.015%로 가정한다.
"""

import pandas as pd

TAX = [("2021-01-01", 0.0023), ("2023-01-01", 0.0020), ("2024-01-01", 0.0018),
       ("2025-01-01", 0.0015), ("2026-01-01", 0.0020)]
COMMISSION = 0.00015


def tax_rate(date) -> float:
    rate = TAX[0][1]
    for start, r in TAX:
        if pd.Timestamp(date) >= pd.Timestamp(start):
            rate = r
    return rate


def net_return(gross: float, sell_date, slip_buy=0.0, slip_sell=0.0, commission=COMMISSION) -> float:
    """비용 반영 전 수익률(gross)을 비용 반영 후로 바꾼다.

    매수: 체결가가 slip_buy 만큼 불리, 수수료.  매도: 체결가가 slip_sell 만큼 불리, 수수료 + 거래세.
    """
    buy = (1 + slip_buy) * (1 + commission)
    sell = (1 + gross) * (1 - slip_sell) * (1 - commission - tax_rate(sell_date))
    return sell / buy - 1
