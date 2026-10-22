"""04-1 일봉 피처 — T일 장 마감 시점까지의 가격·거래대금으로 만든다

규칙 (04-4)
- 직전 추세·변동성은 T-1일까지의 값으로 만든다 (shift(1)). T일 자체의 움직임은 별도 피처로 둔다
- 여러 날에 걸친 비율은 수정주가로, 가격 수준(가격대)은 T일 원주가로 계산한다
- 거래정지일(거래량 0)은 계산에서 뺀다
"""

import numpy as np
import pandas as pd

from limitup.collectors.adjust import adjust

DAILY_FEATURES = [
    "f_ret5", "f_ret20", "f_ret60", "f_vol20", "f_pos60", "f_maxret20", "f_touch60",
    "f_value_log", "f_value_ratio20", "f_gap", "f_low", "f_price_log", "f_kosdaq",
]


def stock_features(s: pd.DataFrame) -> pd.DataFrame:
    """한 종목의 일봉(load_daily 형식)에 일봉 피처 열을 더한다. 모든 창은 과거 방향만 본다."""
    s = adjust(s.sort_values("date"))
    s = s[~s.halted].copy()
    c = s.adj_close
    r1 = c / c.shift(1) - 1
    s["f_ret5"] = c.shift(1) / c.shift(6) - 1                  # T-1일까지 5일 수익률
    s["f_ret20"] = c.shift(1) / c.shift(21) - 1
    s["f_ret60"] = c.shift(1) / c.shift(61) - 1
    s["f_vol20"] = r1.shift(1).rolling(20).std()               # 직전 20일 일간 변동성
    s["f_pos60"] = c.shift(1) / s.adj_high.shift(1).rolling(60).max()   # 60일 고점 대비 위치
    s["f_maxret20"] = r1.shift(1).rolling(20).max()            # 직전 20일 최대 일간 상승
    s["f_touch60"] = (s.lu_touch & ~s.lu_close).astype(int).shift(1).rolling(60, min_periods=1).sum()
    s["f_value_log"] = np.log10(s.value.clip(lower=1))         # T일 거래대금 (원, 로그)
    s["f_value_ratio20"] = s.value / s.value.shift(1).rolling(20).mean()   # 평소 대비 배수
    s["f_gap"] = s.open / s.base - 1                           # T일 시가 갭
    s["f_low"] = s.low / s.base - 1                            # T일 저가 (얼마나 밀렸다 올라왔나)
    s["f_price_log"] = np.log10(s.close)                       # 가격대 (원주가)
    s["f_kosdaq"] = (s.market == "KOSDAQ").astype(int)
    return s


def daily_features(events: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """이벤트(code, date)마다 T일의 일봉 피처를 붙인다."""
    parts = [stock_features(g)[["code", "date", *DAILY_FEATURES]]
             for code, g in df[df.code.isin(events.code.unique())].groupby("code")]
    return events.merge(pd.concat(parts), on=["code", "date"], how="left")
