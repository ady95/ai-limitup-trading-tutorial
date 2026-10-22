import pandas as pd

from limitup.collectors.adjust import adjust


def test_split_is_adjusted_backward():
    # 5:1 액면분할 사례(카카오 2021-04-09 ~ 04-15, 원주가). 분할일 기준가 112,000
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-04-09", "2021-04-12", "2021-04-15", "2021-04-16"]),
        "open": [554000, 558000, 120500, 115500], "high": [561000, 558000, 132500, 120500],
        "low": [551000, 558000, 118000, 115500], "close": [558000, 558000, 120500, 119000],
        "base": [548000, 558000, 112000, 120500],
    })
    adj = adjust(df)
    assert adj.adj_close.round().tolist() == [112000, 112000, 120500, 119000]
    assert adj.adj_factor.iloc[-1] == 1.0
