"""10-4 모의매매 실험 리포트 — 쌓인 체결 기록을 전체·적격·전략 선택으로 나눠 요약한다

    uv run python -m limitup.execution.report

- 전체 주문: 장 마감 단일가 상한가 주문 일반의 관찰 (원기록 그대로)
- 적격 주문: eligible 열이 True 인 주문 — 첫 상한가 조건 오류 등으로 제외한 주문은 exclude_reason 에 이유를 남긴다
- 전략 선택: 적격 주문 중 strategy_pick (점수 1위)
- 원인: 대기열 대용 비율(종가 단일가 거래량 ÷ (거래량 + 15:31 매수 1호가 잔량)), 첫 도달 뒤 잠김 여부별 체결률
- 다음 날 성과: 매도 기록(paper_sell_log.csv), 지연 청산(delayed_exit)은 따로 표시
- 비교: 08-6의 사전 연구용 중단 기준(금액 체결률 50%)과 단순 비교 — 검증된 손익분기점이 아니다
"""

import pandas as pd

from limitup.paths import PROCESSED

FILL_LOG = PROCESSED / "paper_fill_log.csv"
SELL_LOG = PROCESSED / "paper_sell_log.csv"
THRESHOLD = 0.5


def rates(x: pd.DataFrame) -> str:
    if x.empty:
        return "0건"
    amt = x.fill_amt.sum() / x.order_amt.sum()
    return (f"{len(x)}건  금액 체결률 {amt:.2%}  수량 체결률 {x.filled_qty.sum() / x.order_qty.sum():.2%}  "
            f"한 주라도 체결 {(x.filled_qty > 0).sum()}건")


def main():
    if not FILL_LOG.exists():
        print("체결 기록이 없습니다 (09-4 daily close).")
        return
    f = pd.read_csv(FILL_LOG, dtype={"code": str})
    f["order_amt"] = f.order_qty * f.price
    f["fill_amt"] = f.filled_qty * f.price
    f["eligible"] = f["eligible"].astype(str).eq("True") if "eligible" in f else True
    pick = f["strategy_pick"].astype(str).eq("True") if "strategy_pick" in f else pd.Series(False, index=f.index)
    ok = f[f.eligible]
    print(f"실험 {f.date.nunique()}거래일 ({f.date.min()} ~ {f.date.max()})")
    print(f"  전체 주문(상한가 주문 일반의 관찰): {rates(f)}")
    print(f"  적격 주문(첫 상한가 조건 충족):     {rates(ok)}")
    print(f"  적격 중 전략 B 선택(점수 1위):      {rates(f[f.eligible & pick])}")
    excluded = f[~f.eligible]
    if len(excluded):
        print("  제외한 주문:", ", ".join(f"{r.date} {r.code}({r.exclude_reason})" for r in excluded.itertuples()))
    if "provisional" in f and f.provisional.astype(str).eq("True").any():
        print("  잠정 기록(종가 단일가 봉 없음):", ", ".join(f[f.provisional.astype(str).eq("True")].code))
    ok = ok.assign(locked=ok.breaks_after_touch == 0)
    g = ok.groupby("locked").apply(lambda x: pd.Series({"orders": len(x), "fill_amt_rate": x.fill_amt.sum() / x.order_amt.sum()}),
                                   include_groups=False)
    print("  적격 주문의 첫 도달 뒤 잠김 여부별 (True = 끝까지 잠김):")
    print(g.to_string(formatters={"orders": "{:.0f}".format, "fill_amt_rate": "{:.2%}".format}))
    if "bid_qty1_after_close" in f:
        share = f.auction_volume / (f.auction_volume + f.bid_qty1_after_close)
        print(f"  대기열 대용 비율(거래량·잔량 기반, 전체 주문): 중앙값 {share.median():.1%} "
              f"(최소 {share.min():.1%}, 최대 {share.max():.1%})")
    d = f.groupby("date").apply(lambda x: x.fill_amt.sum() / x.order_amt.sum(), include_groups=False)
    print("  날짜별 금액 체결률(전체):", ", ".join(f"{k} {v:.1%}" for k, v in d.items()))
    if SELL_LOG.exists():
        s = pd.read_csv(SELL_LOG, dtype={"code": str, "order_no": str, "order_time": str}).fillna("")
        late = s[s["delayed_exit"].astype(str).eq("True")] if "delayed_exit" in s else s.iloc[0:0]
        print(f"\n매도 기록 {len(s)}건 (다음 날 시가 단일가), 그중 지연 청산 {len(late)}건")
        # 매도분을 직전 체결 기록(매수일 < 매도일)과 짝지어 수수료·세금 전 수익률을 낸다
        buys = f[f.filled_qty > 0][["date", "code", "price", "eligible"]]
        rows = []
        for r in s.itertuples():
            b = buys[(buys.code == r.code) & (buys.date < r.date)].tail(1)
            if len(b):
                rows.append({"매도일": r.date, "code": r.code, "수량": int(r.filled_qty), "매수가": int(b.price.iloc[0]),
                             "매도가": int(r.avg_price), "수익률(비용 전)": f"{int(r.avg_price) / b.price.iloc[0] - 1:+.2%}",
                             "적격": bool(b.eligible.iloc[0]), "지연 청산": str(r.delayed_exit) == "True"})
        print(pd.DataFrame(rows).to_string(index=False) if rows else s.to_string(index=False))
    else:
        print("\n매도 기록 없음 — 다음 거래일 daily open 이후 생깁니다.")
    if ok.empty:
        print(f"\n08-6 연구용 중단 기준(금액 체결률 {THRESHOLD:.0%})과 비교할 적격 주문이 없습니다.")
        return
    amt = ok.fill_amt.sum() / ok.order_amt.sum()
    side = "이상" if amt >= THRESHOLD else "미만"
    print(f"\n08-6 연구용 중단 기준(금액 체결률 {THRESHOLD:.0%}, 검증된 손익분기점 아님)과 단순 비교 — "
          f"적격 주문 {amt:.2%}, 기준 {side}")


if __name__ == "__main__":
    main()
