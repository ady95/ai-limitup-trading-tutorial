"""09-4 하루 실험 일정 — 장 마감 단일가 상한가 매수의 체결률을 모의투자로 잰다

    uv run python -m limitup.execution.daily close      # 15:10 이전에 실행 → 15:20 주문 → 15:31 기록
    uv run python -m limitup.execution.daily open       # 08:50 이전에 실행 → 08:55 시가 매도 → 09:05 기록
    uv run python -m limitup.execution.daily close --dry-run   # 주문 없이 전 과정 점검

close 단계
  1. 15:10 첫 상한가 후보 탐지(09-2)와 시가 갭 예측 점수(09-3)
  2. 15:20:05 장 마감 단일가에 후보 전부 상한가 지정가 매수 (체결률 실험) — 점수 1위는 strategy_pick 표시
  3. 15:31 남은 매수 1호가 잔량 스냅숏, 15:40 체결 내역·분봉 재조회 → data/processed/paper_fill_log.csv 에 추가
open 단계
  4. 08:55 보유분 전량 시장가 매도 (시가 단일가)
  5. 09:05 매도 체결가를 data/processed/paper_sell_log.csv 에 추가
  6. 일봉·지수 증분 수집 (어제 일봉이 확정된 뒤 — 오늘 close 단계의 후보 탐지·점수에 쓴다)
안전 규칙
  - 늦게 시작하면(허용 시각을 넘기면) 주문하지 않는다. 표준 일정이 아닌 결과를 실험 기록에 섞지 않기 위해서다
  - 다음 개장일이 SPECIAL_DAYS 면 close 단계에서 신규 매수를 하지 않는다 (T+1 시가 청산이 어긋나므로)
  - 제때 팔지 못하고 넘긴 보유분은 다음 정규 거래일 open 에서 팔고 지연 청산으로 기록한다
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
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

# 장 운영 시간이 평소와 다른 날 — 고정 시각(08:55·15:20·15:31) 일정이 맞지 않으므로 자동 실행을 멈춘다.
# 거래소 공지(수능일, 연초 개장일 등)를 보고 해마다 갱신한다.
SPECIAL_DAYS = {
    "2026-11-19": "대학수학능력시험일 — 장 운영 시간 변경(공지 확인)",
    "2027-01-04": "연초 개장일 — 10:00 개장",
}

FILL_LOG = PROCESSED / "paper_fill_log.csv"
SCORED = PROCESSED / "today_candidates_scored.csv"   # 10-2 대시보드의 오늘의 후보 탭
SELL_LOG = PROCESSED / "paper_sell_log.csv"
DELAYED = PROCESSED / "paper_orders" / "delayed_exit.json"   # 제때 팔지 못한 보유분 표시
EXPERIMENT_VERSION = "2026-10-25"    # 실행 코드 버전 (live_end 수정 이후)


def is_open_day(kis_real, day=None) -> bool:
    """한국투자증권 휴장일 조회(국내휴장일조회)로 오늘이 개장일인지 확인한다. 평일 공휴일 대비."""
    day = (day or dt.date.today()).strftime("%Y%m%d")
    b = kis_real.get("/uapi/domestic-stock/v1/quotations/chk-holiday", "CTCA0903R",
                     {"BASS_DT": day, "CTX_AREA_NK": "", "CTX_AREA_FK": ""})
    return next((r["opnd_yn"] == "Y" for r in b["output"] if r["bass_dt"] == day), False)


def next_open_day(kis_real, day=None):
    """day 다음 개장일 (휴장일 조회 API는 기준일부터 여러 날을 돌려준다)"""
    nxt = (day or dt.date.today()) + dt.timedelta(days=1)
    b = kis_real.get("/uapi/domestic-stock/v1/quotations/chk-holiday", "CTCA0903R",
                     {"BASS_DT": nxt.strftime("%Y%m%d"), "CTX_AREA_NK": "", "CTX_AREA_FK": ""})
    days = sorted(r["bass_dt"] for r in b["output"] if r["opnd_yn"] == "Y")
    return dt.datetime.strptime(days[0], "%Y%m%d").date() if days else None


def regular_day(day=None) -> bool:
    """SPECIAL_DAYS 에 있는 날이면 이유를 출력하고 False"""
    day = (day or dt.date.today()).isoformat()
    if day in SPECIAL_DAYS:
        print(f"{day} {SPECIAL_DAYS[day]} — 자동 실행하지 않음 (남은 보유분은 다음 정규 거래일 open 에서 지연 청산)")
        return False
    return True


def wait_until(hhmmss: str):
    t = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(hhmmss))
    while dt.datetime.now() < t:
        time.sleep(0.5)


def on_time(start: str, latest: str) -> bool:
    """start 까지 기다린다. 이미 latest 를 넘겼으면(늦은 실행) False — 그 단계의 주문을 내지 않는다."""
    wait_until(start)
    now = dt.datetime.now().time()
    if now > dt.time.fromisoformat(latest):
        print(f"늦은 실행 {now:%H:%M:%S} (허용 {start}~{latest}) — 표준 일정이 아니므로 주문하지 않음", flush=True)
        return False
    return True


def mark_delayed(codes, reason):
    DELAYED.parent.mkdir(parents=True, exist_ok=True)
    DELAYED.write_text(json.dumps({"codes": codes, "reason": reason, "since": dt.date.today().isoformat()},
                                  ensure_ascii=False), encoding="utf-8")


def append(path, df):
    old = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame()
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


def close_phase(dry_run=False, max_orders=5):
    real, paper = KIS("real"), KIS("paper")
    today = dt.date.today()
    if not is_open_day(real):
        print(f"{today} 휴장일 — 실행하지 않음")
        return
    if not regular_day(today):
        return
    nxt = next_open_day(real, today)
    if nxt and nxt.isoformat() in SPECIAL_DAYS:
        print(f"다음 개장일 {nxt} 은 {SPECIAL_DAYS[nxt.isoformat()]} — T+1 시가 청산이 어긋나므로 신규 매수하지 않음")
        return
    if not on_time("15:10:00", "15:14:00"):
        return
    now = limit_up_now(real)
    cands = first_limit_up(now).assign(n_limit_up=len(now))
    if cands.empty:
        print("첫 상한가 후보 없음")
        return
    scored = build_rows(cands, real, today)
    scored["pred_gap"] = gap_model().predict(scored[FEATURES])
    cands = cands.merge(scored[["code", "pred_gap"]], on="code").sort_values("pred_gap", ascending=False)
    cands.assign(date=today.isoformat()).to_csv(SCORED, index=False)
    if len(cands) > max_orders:
        print(f"후보 {len(cands)}종목 중 점수 상위 {max_orders}종목만 주문")
    cands = cands.head(max_orders)
    cash = balance(paper)["cash"]
    amount = min(2_000_000, cash // len(cands))
    print(cands[["code", "upper", "value", "pred_gap"]].to_string(index=False), f"\n종목당 {amount:,}원")

    guard = Guard(paper, max_orders=max_orders, max_amount=2_000_000, max_total=cash)
    if not on_time("15:20:05", "15:28:00"):
        return
    sent = []
    for i, r in enumerate(cands.itertuples()):
        ok, why = guard.allow(r.code, amount)
        if not ok:
            print(r.code, "주문 안 함:", why)
            continue
        qty = amount // r.upper
        if not dry_run:
            guard.reserve(r.code, amount)
        res = {"rt_cd": "dry-run"} if dry_run else buy_limit(paper, r.code, int(r.upper), int(qty))
        if not dry_run:
            guard.record(r.code, amount, res)
        sent.append({"code": r.code, "price": int(r.upper), "order_qty": int(qty), "pred_gap": r.pred_gap,
                     "strategy_pick": i == 0, "rt_cd": res.get("rt_cd"),
                     "sent_at": dt.datetime.now().isoformat(timespec="seconds")})
        print(sent[-1], flush=True)

    # 15:31 남은 매수 1호가 잔량 스냅숏 (대기열 대용 비율의 분모, 09-4)
    wait_until("15:31:00")
    snap_at = dt.datetime.now().isoformat(timespec="seconds")
    bids = {}
    for s in sent:
        q = real.get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", "FHKST01010200",
                     {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": s["code"]})["output1"]
        bids[s["code"]] = int(q["bidp_rsqn1"]) if int(q["bidp1"]) == s["price"] else 0
    # 15:40 체결·분봉 재조회 — 종가 단일가가 연장된 날(04-2)에도 확정된 값을 쓰기 위해
    wait_until("15:40:00")
    checked_at = dt.datetime.now().isoformat(timespec="seconds")
    f = fills(paper, today.strftime("%Y%m%d")) if not dry_run else pd.DataFrame(columns=["code", "filled_qty"])
    rows = []
    for s in sent:
        m = fetch_day(real, s["code"], today.strftime("%Y%m%d"))
        first = m[m.high >= s["price"]].time.min()
        filled = int(f[f.code == s["code"]].filled_qty.astype(int).sum()) if len(f) else 0
        rows.append({"date": today.isoformat(), **s, "filled_qty": filled, "fill_ratio": filled / s["order_qty"],
                     "auction_volume": int(m[m.time >= "153000"].volume.sum()), "day_volume": int(m.volume.sum()),
                     "first_touch": first, "breaks_after_touch": int((m[m.time > first].low < s["price"]).sum()),
                     "closed_at_limit": bool(m.close.iloc[-1] == s["price"]), "order_time": "15:20",
                     "bid_qty1_after_close": bids[s["code"]], "bid_snapshot_at": snap_at, "fills_checked_at": checked_at,
                     "provisional": not (m.time >= "153000").any(),   # 종가 단일가 봉이 아직 없으면 잠정
                     "eligible": True, "exclude_reason": "", "experiment_version": EXPERIMENT_VERSION})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if not dry_run:
        append(FILL_LOG, out)


def refresh_data():
    """일봉·지수 증분 수집을 별도 프로세스로 실행한다 (02-3, 04-3). 실패해도 매매 기록은 이미 끝났다."""
    for module in ("limitup.collectors.daily", "limitup.collectors.index"):
        r = subprocess.run([sys.executable, "-m", module])
        print(module, "완료" if r.returncode == 0 else f"실패 (종료 코드 {r.returncode})", flush=True)


def open_phase(dry_run=False):
    paper = KIS("paper")
    if not is_open_day(KIS("real")):
        print(f"{dt.date.today()} 휴장일 — 실행하지 않음")
        return
    hold = holdings(paper)
    if not regular_day():
        if hold and not dry_run:
            mark_delayed([h["code"] for h in hold], "특별 장 운영일")
        return
    if not on_time("08:55:00", "08:58:30"):
        if hold and not dry_run:
            mark_delayed([h["code"] for h in hold], "늦은 실행")
        if not dry_run:
            refresh_data()
        return
    delayed = json.loads(DELAYED.read_text(encoding="utf-8")) if DELAYED.exists() else {"codes": []}
    for h in hold:
        r = {"rt_cd": "dry-run"} if dry_run else sell_market(paper, h["code"], h["qty"])
        print(h, r.get("rt_cd"), r.get("msg1", ""), flush=True)
    wait_until("09:05:00")
    if hold and not dry_run:
        f = fills(paper, dt.date.today().strftime("%Y%m%d"))
        f = f.assign(date=dt.date.today().isoformat(), delayed_exit=f.code.isin(delayed["codes"]),
                     delay_reason=f.code.map(lambda c: delayed.get("reason", "") if c in delayed["codes"] else ""))
        print(f.to_string(index=False))
        append(SELL_LOG, f)
        DELAYED.unlink(missing_ok=True)
    if not dry_run:
        refresh_data()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["close", "open"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    close_phase(a.dry_run) if a.phase == "close" else open_phase(a.dry_run)
