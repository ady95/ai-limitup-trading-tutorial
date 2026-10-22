"""03-3 ~ 03-5 첫 상한가 이후 다음 거래일 통계와 H1 검증 리포트

    uv run python -m limitup.events.stats

03-2의 detect 로 만든 이벤트와 같은 조건으로 대조군을 만든다.
  close   : 첫 상한가 마감 (분석 대상)
  touch   : 첫 상한가 터치 후 이탈 마감
  control : +20% 이상 올랐지만 상한가에 닿지 않은 날 (같은 제외 조건, 직전 60거래일 상한가 없음)
신뢰구간은 같은 날 이벤트끼리 묶어 날짜 단위로 재표본추출(부트스트랩)한다.
"""

import numpy as np
import pandas as pd

from limitup.events.detect import attach_outcomes, detect, load_daily
from limitup.paths import PROCESSED

REPORT = PROCESSED / "report_ch03.html"
RNG = np.random.default_rng(42)


def control_group(d):
    """+20% 이상 상승 마감했지만 장중에도 상한가에 닿지 않은 날"""
    ok = (d.date >= "2021-01-01") & d.history_ok & (d.prior_lu == 0) & ~d.listing_day \
        & ~d.liquidation & ~d.prev_halted & ~d.halted
    ctl = d[ok & ~d.lu_touch & (d.close >= d.base * 1.2)].copy()
    ctl["group"] = "control"
    return ctl


def cluster_ci(df, col, n_boot=2000):
    """날짜 단위 부트스트랩 95% 신뢰구간 (평균)"""
    by_day = df.groupby("date")[col].agg(["sum", "count"])
    s, c = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    pick = RNG.integers(0, len(s), size=(n_boot, len(s)))
    means = s[pick].sum(1) / c[pick].sum(1)
    return np.percentile(means, [2.5, 97.5])


def pct(x):
    return f"{x * 100:.1f}%"


def section_33(ev, d):
    print("\n[03-3] 다음 거래일 상한가 도달·마감 비율")
    traded = d[(d.date >= "2021-01-01") & ~d.halted & ~d.listing_day & ~d.liquidation]
    print(f"  기준선: 아무 종목·아무 날의 상한가 터치 {pct(traded.lu_touch.mean())}, "
          f"마감 {pct(traded.lu_close.mean())} ({len(traded):,}행)")
    t = ev[ev.has_t1].groupby("group").agg(
        n=("code", "size"), t1_touch=("t1_lu_touch", "mean"), t1_close=("t1_lu_close", "mean"),
        p3=("p3", "mean"))
    print(t.to_string(formatters={c: pct for c in ["t1_touch", "t1_close", "p3"]}))
    c = ev[(ev.group == "close") & ev.has_t1]
    for col in ("t1_lu_touch", "t1_lu_close"):
        lo, hi = cluster_ci(c.assign(**{col: c[col].astype(float)}), col)
        print(f"  마감 집단 {col} 95% 신뢰구간: {pct(lo)} ~ {pct(hi)}")
    by = c.groupby([c.date.dt.year.rename("year")]).agg(
        n=("code", "size"), t1_touch=("t1_lu_touch", "mean"), t1_close=("t1_lu_close", "mean"))
    print(by.to_string(formatters={"t1_touch": pct, "t1_close": pct}))
    bm = c.groupby("market").agg(n=("code", "size"), t1_touch=("t1_lu_touch", "mean"),
                                 t1_close=("t1_lu_close", "mean"))
    print(bm.to_string(formatters={"t1_touch": pct, "t1_close": pct}))
    per_day = c.groupby("date").size()
    print(f"  이벤트가 있는 날 {len(per_day):,}일, 하루 평균 {per_day.mean():.2f}건, 최대 {per_day.max()}건, "
          f"하루 5건 이상인 날의 이벤트 비중 {per_day[per_day >= 5].sum() / per_day.sum():.1%}")


def section_34(ev):
    print("\n[03-4] 다음 날 시가·고가·종가 수익률 (T+1 기준가 = T일 종가 대비)")
    c = ev[(ev.group == "close") & ev.has_t1]
    cols = {"t1_open_ret": "시가", "t1_high_ret": "고가", "t1_low_ret": "저가", "t1_close_ret": "종가",
            "t1_oc_ret": "시가→종가", "t1_mfe": "시가 대비 고가(MFE)", "t1_mae": "시가 대비 저가(MAE)"}
    q = c[list(cols)].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T
    q.insert(0, "mean", c[list(cols)].mean())
    q.insert(1, "plus", (c[list(cols)] > 0).mean())
    q.index = list(cols.values())
    print(q.to_string(float_format=lambda x: f"{x * 100:.1f}%"))
    locked = (c.t1_open == c.t1_upper)
    print(f"  시가가 상한가(시초가 상한가): {locked.sum()}건 ({locked.mean():.1%}) — 이 중 상한가 마감 "
          f"{c[locked].t1_lu_close.mean():.1%}")
    bins = [-1, -0.05, 0, 0.05, 0.10, 0.20, 1]
    labels = ["-5% 미만", "-5~0%", "0~5%", "5~10%", "10~20%", "20% 이상(상한가 제외)", "상한가 시가"]
    gap = pd.cut(c.t1_open_ret, bins, labels=labels[:-1], right=False)
    gap = gap.cat.add_categories(labels[-1])
    gap[locked] = labels[-1]
    g = c.groupby(gap, observed=True).agg(
        n=("code", "size"), close_ret=("t1_close_ret", "mean"), oc_ret=("t1_oc_ret", "mean"),
        oc_plus=("t1_oc_ret", lambda x: (x > 0).mean()), t1_close=("t1_lu_close", "mean"))
    print(g.to_string(formatters={"close_ret": pct, "oc_ret": pct, "oc_plus": pct, "t1_close": pct}))


def section_35(ev):
    print("\n[03-5] H1 검증: 집단별 T+1 결과 (평균과 날짜 단위 95% 신뢰구간)")
    rows = []
    for grp, g in ev[ev.has_t1].groupby("group"):
        for col in ("t1_close_ret", "t1_oc_ret", "t1_open_ret"):
            lo, hi = cluster_ci(g, col)
            rows.append({"group": grp, "metric": col, "n": len(g), "mean": g[col].mean(),
                         "median": g[col].median(), "ci_lo": lo, "ci_hi": hi,
                         "plus": (g[col] > 0).mean()})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, formatters={c: pct for c in ["mean", "median", "ci_lo", "ci_hi", "plus"]}))
    return t


def report(ev, h1):
    """Plotly HTML 리포트"""
    import plotly.express as px
    import plotly.io as pio

    c = ev[ev.has_t1]
    figs = [
        px.histogram(c, x="t1_close_ret", color="group", barmode="overlay", nbins=80,
                     histnorm="probability", title="T+1 종가 수익률 분포 (집단별)"),
        px.bar(c[c.group == "close"].groupby(c.date.dt.year).t1_lu_close.mean().reset_index(),
               x="date", y="t1_lu_close", title="연도별 T+1 상한가 마감 비율 (마감 집단)"),
        px.scatter(c[c.group == "close"], x="t1_open_ret", y="t1_close_ret", opacity=0.3,
                   title="T+1 시가 갭과 종가 수익률"),
    ]
    html = "".join(pio.to_html(f, full_html=False, include_plotlyjs=(i == 0)) for i, f in enumerate(figs))
    table = h1.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    REPORT.write_text(f"<html><meta charset='utf-8'><body><h2>03장 첫 상한가 통계 리포트</h2>"
                      f"{table}{html}</body></html>", encoding="utf-8")
    print(f"\n리포트 → {REPORT}")


def main():
    df = load_daily()
    ev, d = detect(df, 60)
    ev = attach_outcomes(pd.concat([ev, control_group(d)]), d)
    ev.to_parquet(PROCESSED / "events_with_control.parquet", index=False)
    section_33(ev, d)
    section_34(ev)
    h1 = section_35(ev)
    report(ev, h1)


if __name__ == "__main__":
    main()
