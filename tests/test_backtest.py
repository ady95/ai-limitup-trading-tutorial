import pandas as pd

from limitup.backtest.costs import net_return, tax_rate
from limitup.backtest.engine import Engine, Order


def test_tax_by_year():
    assert tax_rate("2024-06-01") == 0.0018
    assert tax_rate("2025-06-01") == 0.0015
    assert tax_rate("2026-06-01") == 0.0020


def test_engine_round_trip_matches_net_return():
    t0, t1 = pd.Timestamp("2026-03-02 15:30"), pd.Timestamp("2026-03-03 09:00")
    e = Engine(100_000_000)
    e.submit(Order("000000", t0, 10_000, t1, 11_000, 10_000_000))
    e.submit(Order("000001", t0, 10_000, t1, 11_000, 10_000_000, filled=False))
    s = e.run().summary()
    assert s["orders"] == 2 and s["trades"] == 1
    assert abs(s["mean_ret"] - net_return(0.10, t1)) < 1e-9
