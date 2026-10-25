"""10-4 모의매매 실험 리포트 — 쌓인 체결 기록으로 08-6의 판단 기준을 점검한다

    uv run python -m limitup.execution.report

- 체결률: 주문 수량·금액 기준, 날짜별, 전략 B 선택(strategy_pick)만 따로
- 원인: 대기열 대용 비율(종가 단일가 거래량 ÷ (거래량 + 15:31 매수 1호가 잔량)), 첫 도달 뒤 잠김 여부별 체결률
- 다음 날 성과: 매도 기록(paper_sell_log.csv)이 있으면 체결분의 실제 손익
- 판정: 08-6의 실전 검토 기준(금액 체결률 50%)과 비교
"""

import pandas as pd

from limitup.paths import PROCESSED

FILL_LOG = PROCESSED / "paper_fill_log.csv"
SELL_LOG = PROCESSED / "paper_sell_log.csv"
THRESHOLD = 0.5


def main():
    if not FILL_LOG.exists():
        print("체결 기록이 없습니다 (09-4 daily close).")
        return
    f = pd.read_csv(FILL_LOG, dtype={"code": str})
    f["order_amt"] = f.order_qty * f.price
    f["fill_amt"] = f.filled_qty * f.price
    amt = f.fill_amt.sum() / f.order_amt.sum()
    print(f"실험 {f.date.nunique()}거래일 ({f.date.min()} ~ {f.date.max()}), 주문 {len(f)}건")
    print(f"  금액 체결률 {amt:.2%}  수량 체결률 {f.filled_qty.sum() / f.order_qty.sum():.2%}  "
          f"한 주라도 체결된 주문 {(f.filled_qty > 0).mean():.0%}")
    if "strategy_pick" in f:
        p = f[f.strategy_pick.astype(str) == "True"]
        if len(p):
            print(f"  전략 B 선택 종목만: {len(p)}건, 금액 체결률 {p.fill_amt.sum() / p.order_amt.sum():.2%}")
    f["locked"] = f.breaks_after_touch == 0
    g = f.groupby("locked").apply(lambda x: pd.Series({"orders": len(x), "fill_amt_rate": x.fill_amt.sum() / x.order_amt.sum()}),
                                  include_groups=False)
    print("  첫 도달 뒤 잠김 여부별 (True = 끝까지 잠김):")
    print(g.to_string(formatters={"orders": "{:.0f}".format, "fill_amt_rate": "{:.2%}".format}))
    if "bid_qty1_after_close" in f:
        share = f.auction_volume / (f.auction_volume + f.bid_qty1_after_close)
        print(f"  대기열 대용 비율(거래량·잔량 기반): 중앙값 {share.median():.1%} (최소 {share.min():.1%}, 최대 {share.max():.1%})")
    d = f.groupby("date").apply(lambda x: x.fill_amt.sum() / x.order_amt.sum(), include_groups=False)
    print("  날짜별 금액 체결률:", ", ".join(f"{k} {v:.1%}" for k, v in d.items()))
    if SELL_LOG.exists():
        s = pd.read_csv(SELL_LOG, dtype={"code": str})
        print(f"\n매도 기록 {len(s)}건 (다음 날 시가 단일가)")
        print(s.to_string(index=False))
    else:
        print("\n매도 기록 없음 — 다음 거래일 daily open 이후 생깁니다.")
    verdict = "충족" if amt >= THRESHOLD else "미충족"
    print(f"\n08-6 실전 검토 기준(금액 체결률 {THRESHOLD:.0%} 이상): {verdict} — 현재 {amt:.2%}")


if __name__ == "__main__":
    main()
