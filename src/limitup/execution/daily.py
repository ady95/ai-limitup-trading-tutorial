"""09-4 하루 실험 일정 — 장 마감 단일가 상한가 매수의 체결률을 모의투자로 잰다

    uv run python -m limitup.execution.daily close      # 15:10 이전에 실행 → 15:20 주문 → 15:31 기록
    uv run python -m limitup.execution.daily open       # 08:50 이전에 실행 → 08:55 시가 매도 → 09:05 기록
    uv run python -m limitup.execution.daily close --dry-run   # 주문 없이 전 과정 점검

close 단계
  1. 15:10 첫 상한가 후보 탐지(09-2)와 시가 갭 예측 점수(09-3)
  2. 15:20:05 장 마감 단일가에 후보 전부 상한가 지정가 매수 (체결률 실험) — 점수 1위는 strategy_pick 표시
  3. 15:31 체결 내역, 장 마감 단일가 거래량, 남은 상한가 대기열을 data/processed/paper_fill_log.csv 에 추가
open 단계
  4. 08:55 보유분 전량 시장가 매도 (시가 단일가)
  5. 09:05 매도 체결가를 data/processed/paper_sell_log.csv 에 추가
"""

import argparse
import datetime as dt
import time

import pandas as pd

from limitup.collectors.kis import KIS
from limitup.collectors.minute import fetch_day
from limitup.execution.candidates import first_limit_up, limit_up_now
from limitup.execution.guard import Guard
from limitup.execution.live_score import build_rows, gap_model
from limitup.execution.paper import balance, buy_limit, fills, holdings, sell_market
from limitup.models.common import FEATURES
from limitup.paths import PROCESSED

FILL_LOG = PROCESSED / "paper_fill_log.csv"
SELL_LOG = PROCESSED / "paper_sell_log.csv"


def wait_until(hhmmss: str):
    t = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(hhmmss))
    while dt.datetime.now() < t:
        time.sleep(0.5)


def append(path, df):
    old = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame()
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


def close_phase(dry_run=False, max_orders=5):
    real, paper = KIS("real"), KIS("paper")
    today = dt.date.today()
    wait_until("15:10:00")
    now = limit_up_now(real)
    cands = first_limit_up(now).assign(n_limit_up=len(now))
    if cands.empty:
        print("첫 상한가 후보 없음")
        return
    scored = build_rows(cands, real, today)
    scored["pred_gap"] = gap_model().predict(scored[FEATURES])
    cands = cands.merge(scored[["code", "pred_gap"]], on="code").sort_values("pred_gap", ascending=False)
    cands = cands.head(max_orders)
    cash = balance(paper)["cash"]
    amount = min(2_000_000, cash // len(cands))
    print(cands[["code", "upper", "value", "pred_gap"]].to_string(index=False), f"\n종목당 {amount:,}원")

    guard = Guard(paper, max_orders=max_orders, max_amount=2_000_000, max_total=cash)
    wait_until("15:20:05")
    sent = []
    for i, r in enumerate(cands.itertuples()):
        ok, why = guard.allow(r.code, amount)
        if not ok:
            print(r.code, "주문 안 함:", why)
            continue
        qty = amount // r.upper
        res = {"rt_cd": "dry-run"} if dry_run else buy_limit(paper, r.code, int(r.upper), int(qty))
        if not dry_run:
            guard.record(r.code, amount, res)
        sent.append({"code": r.code, "price": int(r.upper), "order_qty": int(qty), "pred_gap": r.pred_gap,
                     "strategy_pick": i == 0, "rt_cd": res.get("rt_cd")})
        print(sent[-1], flush=True)

    wait_until("15:31:00")
    f = fills(paper, today.strftime("%Y%m%d")) if not dry_run else pd.DataFrame(columns=["code", "filled_qty"])
    rows = []
    for s in sent:
        m = fetch_day(real, s["code"], today.strftime("%Y%m%d"))
        q = real.get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", "FHKST01010200",
                     {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": s["code"]})["output1"]
        first = m[m.high >= s["price"]].time.min()
        filled = int(f[f.code == s["code"]].filled_qty.astype(int).sum()) if len(f) else 0
        rows.append({"date": today.isoformat(), **s, "filled_qty": filled, "fill_ratio": filled / s["order_qty"],
                     "auction_volume": int(m[m.time >= "153000"].volume.sum()), "day_volume": int(m.volume.sum()),
                     "first_touch": first, "breaks_after_touch": int((m[m.time > first].low < s["price"]).sum()),
                     "closed_at_limit": bool(m.close.iloc[-1] == s["price"]), "order_time": "15:20",
                     "bid_qty1_after_close": int(q["bidp_rsqn1"]) if int(q["bidp1"]) == s["price"] else 0})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if not dry_run:
        append(FILL_LOG, out)


def open_phase(dry_run=False):
    paper = KIS("paper")
    wait_until("08:55:00")
    hold = holdings(paper)
    for h in hold:
        r = {"rt_cd": "dry-run"} if dry_run else sell_market(paper, h["code"], h["qty"])
        print(h, r.get("rt_cd"), r.get("msg1", ""), flush=True)
    wait_until("09:05:00")
    if dry_run or not hold:
        return
    f = fills(paper, dt.date.today().strftime("%Y%m%d"))
    f = f.assign(date=dt.date.today().isoformat())
    print(f.to_string(index=False))
    append(SELL_LOG, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["close", "open"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    close_phase(a.dry_run) if a.phase == "close" else open_phase(a.dry_run)
