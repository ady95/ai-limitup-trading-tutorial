"""07장 전략 설계 — 설계 기간(2022~2024, 워크포워드 점수가 있는 기간)의 비용 반영 전 성과

    uv run python -m limitup.strategies.design

7장의 숫자는 체결이 원하는 가격에 된다고 가정한 '낙관적 상한선'이다.
체결 가능성·슬리피지·비용은 8장에서 반영하고, 평가 기간(2025~)도 8장에서 처음 쓴다.

일봉으로 손절·익절을 판정할 때는 같은 날 둘 다 닿았으면 손절이 먼저라고 본다(보수적 가정).
"""

import numpy as np
import pandas as pd

from limitup.events.detect import detect, load_daily
from limitup.models.common import load
from limitup.paths import PROCESSED
from limitup.strategies.scores import SCORES

DESIGN = ("2022-01-01", "2025-01-01")
pct = "{:+.2%}".format


def summary(ret: pd.Series) -> dict:
    return {"n": len(ret), "mean": ret.mean(), "median": ret.median(), "win": (ret > 0).mean(),
            "p10": ret.quantile(0.1), "worst": ret.min()}


def show(rows: dict, title: str):
    t = pd.DataFrame({k: summary(v) for k, v in rows.items()}).T
    print(f"\n[{title}]")
    print(t.to_string(formatters={"n": "{:,.0f}".format, "mean": pct, "median": pct, "win": "{:.1%}".format,
                                  "p10": pct, "worst": pct}))


def design_set() -> pd.DataFrame:
    ds = load()
    sc = pd.read_parquet(SCORES)
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    t1 = ev[ev.group == "close"][["code", "date", "t1_open", "t1_high", "t1_low", "t1_close", "t1_upper"]]
    d = ds.merge(sc, on=["code", "date"]).merge(t1, on=["code", "date"])
    return d[(d.date >= DESIGN[0]) & (d.date < DESIGN[1])].copy()


def top_k(d: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    return d.sort_values(score, ascending=False).groupby("date").head(k)


# ---------- 07-1 종목 선택 ----------
def selection(d):
    print(f"설계 기간 {d.date.min():%Y-%m-%d} ~ {d.date.max():%Y-%m-%d}, 첫 상한가 {len(d):,}건, 이벤트가 있는 날 {d.date.nunique():,}일")
    rows = {}
    for score in ("p2", "pred_gap"):
        for k in (1, 3):
            s = top_k(d, score, k)
            rows[f"{score} 하루 상위 {k}"] = s
    rows["전체"] = d
    t = pd.DataFrame({name: {"n": len(s), "연속 상한가": s.y_close.mean(), "T+1 시가 갭": s.y_ret_open.mean(),
                             "T+1 시가→종가": s.y_ret_oc.mean(), "T+1 종가": s.y_ret_close.mean()}
                      for name, s in rows.items()}).T
    print("\n[07-1 선택 방식별 결과 — 비용 반영 전]")
    print(t.to_string(formatters={"n": "{:,.0f}".format, "연속 상한가": "{:.1%}".format,
                                  "T+1 시가 갭": pct, "T+1 시가→종가": pct, "T+1 종가": pct}))
    v = d.f_value_log
    print(f"\n거래대금 10억 원 미만 첫 상한가: {(v < 9).mean():.1%}, 30억 원 미만: {(v < np.log10(3e9)).mean():.1%}")


# ---------- 07-2 전략 C: T+1 시가 매수 ----------
def exit_with_stops(entry, high, low, close, upper, tp, sl):
    """시가 매수 후 같은 날 손절(sl)·익절(tp) 판정. 둘 다 닿으면 손절이 먼저(보수적)."""
    tp_px = np.minimum(entry * (1 + tp), upper) if tp else np.inf
    sl_px = entry * (1 - sl) if sl else -np.inf
    hit_sl = low <= sl_px
    hit_tp = high >= tp_px
    out = np.where(hit_sl, sl_px, np.where(hit_tp, tp_px, close))
    return out / entry - 1


def strategy_c(d):
    e = d[d.t1_open < d.t1_upper]                             # 시가가 상한가면 살 수 없음
    print(f"\n[07-2 전략 C] T+1 시가가 상한가라 살 수 없던 이벤트: {len(d) - len(e):,}건 ({1 - len(e) / len(d):.1%})")
    rows = {"전체 (종가 청산)": e.t1_close / e.t1_open - 1}
    for tp, sl in [(0.10, 0.05), (0.05, 0.03), (0.15, 0.07)]:
        rows[f"익절 {tp:.0%} / 손절 {sl:.0%}"] = pd.Series(exit_with_stops(
            e.t1_open, e.t1_high, e.t1_low, e.t1_close, e.t1_upper, tp, sl), index=e.index)
    for name, k in [("p2 하루 1위", 1)]:
        s = top_k(e, "p2", k)
        rows[f"{name} (종가 청산)"] = s.t1_close / s.t1_open - 1
    s = top_k(e, "pred_oc", 1)
    rows["장중 수익 예측 하루 1위"] = s.t1_close / s.t1_open - 1
    small_gap = e[e.y_ret_open <= 0.02]
    rows["시가 갭 +2% 이하만"] = small_gap.t1_close / small_gap.t1_open - 1
    show(rows, "07-2 전략 C: T+1 시가 매수 — 비용 반영 전")


# ---------- 07-3 전략 B: T일 상한가 매수 → T+1 청산 ----------
def strategy_b(d):
    rows = {"전체 → T+1 시가 매도": d.y_ret_open, "전체 → T+1 종가 매도": d.y_ret_close}
    for score in ("p2", "pred_gap"):
        s = top_k(d, score, 1)
        rows[f"{score} 하루 1위 → T+1 시가"] = s.y_ret_open
    s = top_k(d, "pred_gap", 1)
    rows["pred_gap 하루 1위 → T+1 종가"] = s.y_ret_close
    show(rows, "07-3 전략 B: T일 상한가에 매수했다고 가정 — 체결 무시, 비용 반영 전 (낙관적 상한선)")


# ---------- 07-4 전략 A: 상한가 도달 전 +25%에서 선제 매수 ----------
def strategy_a(trigger=0.25):
    df = load_daily()
    _, d = detect(df, 60)
    d = d.sort_values(["code", "date"])
    g = d.groupby("code")
    d["n_open"], d["n_base"] = g.open.shift(-1), g.base.shift(-1)
    d["n_close"], d["n_halted"] = g.close.shift(-1), g.halted.shift(-1)
    ok = (d.date >= DESIGN[0]) & (d.date < DESIGN[1]) & d.history_ok & (d.prior_lu == 0) \
        & ~d.listing_day & ~d.liquidation & ~d.prev_halted & ~d.halted & d.n_open.notna() & (d.n_halted == False)
    trig_px = d.base * (1 + trigger)
    cand = d[ok & (d.high >= trig_px)].copy()
    cand["entry"] = np.maximum(cand.open, trig_px[cand.index])            # 시가가 이미 넘었으면 시가에
    cand = cand[cand.entry < cand.upper]                                   # 상한가에 바로 산 경우 제외
    cand["outcome"] = np.select([cand.lu_close, cand.lu_touch], ["상한가 마감", "터치 후 이탈"], "도달 못 함")
    cand["ret_close"] = cand.close / cand.entry - 1
    cand["ret_t1open"] = cand.n_open / cand.entry * (cand.close / cand.n_base) - 1   # T+1 기준가 보정
    print(f"\n[07-4 전략 A] 설계 기간에 기준가 +{trigger:.0%}를 넘은 첫 급등 {len(cand):,}건")
    t = cand.groupby("outcome").agg(n=("code", "size"), ret_close=("ret_close", "mean"),
                                    ret_t1open=("ret_t1open", "mean"))
    t["share"] = t.n / t.n.sum()
    print(t.to_string(formatters={"ret_close": pct, "ret_t1open": pct, "share": "{:.1%}".format}))
    show({"+25% 매수 → T일 종가 매도": cand.ret_close, "+25% 매수 → T+1 시가 매도": cand.ret_t1open},
         "07-4 전략 A — 비용 반영 전")


# ---------- 07-5 제외 조건 ----------
def exclusions(d):
    from limitup.llm.features import llm_features
    x = llm_features(d)
    risk = x.l_group.isin(["risk"])
    rows = {"전체 → T+1 시가 (B)": x.y_ret_open, "위험 재료 제외 → T+1 시가": x[~risk].y_ret_open,
            "위험 재료만 → T+1 시가": x[risk].y_ret_open,
            "거래대금 10억 원 이상 → T+1 시가": x[x.f_value_log >= 9].y_ret_open}
    show(rows, "07-5 제외 조건의 효과 (전략 B 기준, 비용 반영 전)")
    print(f"위험 재료(소송·시장경보) 이벤트 {risk.sum()}건, 연속 상한가 {x[risk].y_close.mean():.1%} (그 밖 {x[~risk].y_close.mean():.1%})")


def main():
    d = design_set()
    selection(d)
    strategy_c(d)
    strategy_b(d)
    strategy_a()
    exclusions(d)


if __name__ == "__main__":
    main()
