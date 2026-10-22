"""08장 백테스트 실행 — 평가 기간(2025-01~)에 7장의 규칙을 처음 적용한다

    uv run python -m limitup.backtest.run

08-2 슬리피지, 08-3 체결 모형(대기열), 08-4 비용, 08-5 기간·국면별 성과, 08-6 스트레스 테스트
"""

import numpy as np
import pandas as pd

from limitup.backtest import plans
from limitup.backtest.costs import net_return
from limitup.backtest.engine import Engine
from limitup.llm.features import llm_features
from limitup.models.common import load
from limitup.paths import INDEX_FILE, PROCESSED
from limitup.strategies.rules import load_config, select
from limitup.strategies.scores import SCORES

CAPITAL = 100_000_000
TEST_START, MINUTE_START = "2025-01-01", "2025-10-14"
pct = "{:+.2%}".format


def data():
    ds = llm_features(load())
    sc = pd.read_parquet(SCORES)
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    t1 = ev[ev.group == "close"][["code", "date", "t1_date", "t1_open", "t1_high", "t1_low", "t1_close",
                                  "t1_upper", "t1_base", "close", "t2_date"]]
    d = ds.merge(sc, on=["code", "date"]).merge(t1, on=["code", "date"])
    return d[d.date >= TEST_START].copy(), ev


def run(orders):
    e = Engine(CAPITAL)
    for o in orders:
        e.submit(o)
    return e.run()


def line(name, s):
    if not s.get("trades"):
        return f"  {name:34s} 주문 {s['orders']:>4}  체결 0"
    return (f"  {name:34s} 주문 {s['orders']:>4}  체결 {s['trades']:>4} ({s['fill_rate']:.0%})  평균 {pct(s['mean_ret'])}  "
            f"승률 {s['win_rate']:.1%}  손익비 {s['payoff']:.2f}  누적 {s['total_return']:+.1%}  MDD {s['mdd']:.1%}")


def sec_slippage(test, cfg):
    print("\n[08-2 슬리피지 — 수수료·세금 포함]")
    b, c = select(test, cfg, "B"), select(test, cfg, "C")
    rc = cfg["strategy"]["C"]
    for s in (0.0, 0.001, 0.003, 0.005):
        print(line(f"B-마감(낙관 체결) 매도 슬리피지 {s:.1%}", run(plans.b_close_orders(b, "optimistic", s)).summary()))
    for s in (0.0, 0.001, 0.003, 0.005):
        print(line(f"C 매수·매도 슬리피지 {s:.1%}", run(plans.c_orders(c, rc["take_profit"], rc["stop_loss"], s)).summary()))


def sec_queue(test, ev):
    print("\n[08-3 체결 모형]")
    b = select(test, load_config(), "B")
    print(line("B-마감 낙관(항상 체결)", run(plans.b_close_orders(b, "optimistic")).summary()))
    print(line("B-마감 엄격(확인 불가 → 미체결)", run(plans.b_close_orders(b, "strict")).summary()))
    w = ev[ev.group.isin(["close", "touch"]) & (ev.date >= MINUTE_START) & ev.has_t1]
    cand = plans.b_touch_candidates(w, 0)
    c = cand[cand.group == "close"]
    print(f"\n  분봉 기간 첫 상한가 도달 {len(cand):,}건 (마감 {len(c)}, 터치 후 이탈 {len(cand) - len(c)})")
    print(f"  마감 집단 중 첫 도달 뒤 상한가 아래 체결이 없던(끝까지 잠긴) 비율 {1 - c.filled.mean():.1%}")
    g = c.groupby("filled").agg(n=("code", "size"), t1_open=("t1_open_ret", "mean"), t1_close_lu=("t1_lu_close", "mean"))
    print("  마감 집단: 대기열 체결 여부별 T+1 결과 (체결 = 상한가가 한 번이라도 풀림)")
    print(g.to_string(formatters={"t1_open": pct, "t1_close_lu": "{:.1%}".format}))
    for delay in (0, 5, 30):
        cd = plans.b_touch_candidates(w, delay) if delay else cand
        s = run(plans.b_touch_orders(cd, 10_000_000)).summary()
        print(line(f"B-터치 주문 지연 {delay}분, T+1 시가 매도", s))
    # 장 마감 때 상한가가 아니면 그날 종가에 판다 (장 마감 시점에 알 수 있는 규칙)
    first = cand.sort_values("touch_time").groupby("date").head(1)
    f = first[first.filled]
    gross = np.where(f.group == "close", f.t1_open / f.t1_base * (f.close / f.upper) - 1, f.close / f.upper - 1)
    sell_date = np.where(f.group == "close", f.t1_date, f.date)
    net = [net_return(g_, d_) for g_, d_ in zip(gross, sell_date)]
    print(f"  B-터치 + 마감 때 상한가 아니면 종가 매도: 주문 {len(first)} 체결 {len(f)}  평균 {pct(np.mean(net))}  "
          f"승률 {np.mean(np.array(net) > 0):.1%}  (마감 {int((f.group == 'close').sum())}건 / 이탈 {int((f.group == 'touch').sum())}건)")
    return cand


def sec_costs(test, cfg):
    print("\n[08-4 비용: 비용 반영 전 → 후 (슬리피지 0.1%)]")
    b = select(test, cfg, "B")
    gross = b.t1_open / b.t1_base - 1
    net = [net_return(g, d, slip_sell=0.001) for g, d in zip(gross, b.t1_date)]
    print(f"  B-마감(낙관): 거래당 {pct(gross.mean())} → {pct(np.mean(net))}  (차이 {np.mean(net) - gross.mean():+.2%}p)")
    c = select(test, cfg, "C")
    from limitup.strategies.design import exit_with_stops
    rc = cfg["strategy"]["C"]
    gc = exit_with_stops(c.t1_open, c.t1_high, c.t1_low, c.t1_close, c.t1_upper, rc["take_profit"], rc["stop_loss"])
    nc = [net_return(g, d, 0.001, 0.001) for g, d in zip(gc, c.t1_date)]
    print(f"  C            : 거래당 {pct(np.mean(gc))} → {pct(np.mean(nc))}  (차이 {np.mean(nc) - np.mean(gc):+.2%}p)")
    print("  세율: 2025년 0.15%, 2026년 0.20%, 수수료 편도 0.015% 가정")


def sec_periods(test, cfg):
    print("\n[08-5 기간·국면별 성과 — B-마감 낙관 체결, 슬리피지 0.1%]")
    b = select(test, cfg, "B").copy()
    b["net"] = [net_return(g, d, slip_sell=0.001) for g, d in zip(b.t1_open / b.t1_base - 1, b.t1_date)]
    b["half"] = b.date.dt.year.astype(str) + np.where(b.date.dt.month <= 6, "H1", "H2")
    ix = pd.read_parquet(INDEX_FILE)
    k = ix[ix["index"] == "KOSDAQ"].set_index("date").close
    trend = (k / k.shift(20) - 1).rename("kq20")
    b = b.merge(trend, left_on="date", right_index=True, how="left")
    b["regime"] = pd.cut(b.kq20, [-1, -0.03, 0.03, 1], labels=["하락(-3% 미만)", "보합", "상승(+3% 초과)"])
    for col in ("half", "regime"):
        t = b.groupby(col, observed=True).net.agg(n="size", mean="mean", win=lambda x: (x > 0).mean(), worst="min")
        print(t.to_string(formatters={"mean": pct, "win": "{:.1%}".format, "worst": pct}))
    c = select(test, cfg, "C")
    rc = cfg["strategy"]["C"]
    print(line("C (슬리피지 0.1%)", run(plans.c_orders(c, rc["take_profit"], rc["stop_loss"], 0.001)).summary()))
    print(line("B-마감 낙관 (슬리피지 0.1%)", run(plans.b_close_orders(select(test, cfg, "B"), "optimistic", 0.001)).summary()))
    return b


def sec_stress(b, cand):
    print("\n[08-6 스트레스 테스트 — B-마감 낙관 체결 기준]")
    r = b.net.sort_values(ascending=False)
    print(f"  상위 10% 거래를 빼면: 평균 {pct(r.iloc[len(r) // 10:].mean())} (전체 {pct(r.mean())})")
    print(f"  최악 5건: {', '.join(pct(x) for x in r.tail(5))}")
    streak = (b.sort_values("date").net <= 0).astype(int)
    runs = streak.groupby((streak != streak.shift()).cumsum()).sum()
    print(f"  최장 연속 손실 {int(runs.max())}회")
    for share in (0.5, 0.3):
        # 무작위가 아니라 '잠긴 종목부터 체결 안 됨' — 시가 갭이 큰 순으로 체결을 빼는 역선택 가정
        keep = b.sort_values("net").head(int(len(b) * share))
        print(f"  체결률 {share:.0%}이고 수익이 큰 거래부터 체결되지 않는다면: 평균 {pct(keep.net.mean())}")
    halt = b.t2_date.isna() | (b.t1_date.isna())
    print(f"  T+1 결과가 없는(거래정지 등) 선택 종목: {int(halt.sum())}건")


def main():
    cfg = load_config()
    test, ev = data()
    print(f"평가 기간 {test.date.min():%Y-%m-%d} ~ {test.date.max():%Y-%m-%d}, 첫 상한가 {len(test):,}건")
    sec_slippage(test, cfg)
    cand = sec_queue(test, ev)
    sec_costs(test, cfg)
    b = sec_periods(test, cfg)
    sec_stress(b, cand)


if __name__ == "__main__":
    main()
