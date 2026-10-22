"""02-5 기준가격으로 수정주가 만들기

권리락·주식배당·분할·병합 같은 사건이 있는 날은 기준가격이 전일 종가와 다르다.
그 비율(기준가 / 전일 종가)을 사건 이전의 모든 가격에 곱하면 수정주가가 된다.
현금배당은 기준가격을 바꾸지 않으므로 자동으로 제외된다.

    from limitup.collectors.adjust import adjust
    adj = adjust(df)          # df: 한 종목의 일봉 (date, open, high, low, close, base)
"""

import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    """원주가 일봉에 수정주가 열(adj_open ... adj_close)과 누적 수정계수(adj_factor)를 더한다."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    ratio = (df.base / df.close.shift(1)).fillna(1.0)
    # 사건일의 비율은 그 전날까지의 가격에 적용된다 → 뒤에서부터 누적곱한 뒤 한 칸 당긴다
    df["adj_factor"] = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for c in PRICE_COLS:
        df[f"adj_{c}"] = df[c] * df.adj_factor
    return df
