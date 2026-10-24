"""09-4 모의투자 주문·체결 조회 (한국투자증권 모의투자 계좌)

    uv run python -m limitup.execution.paper balance
    uv run python -m limitup.execution.paper buy-close data/processed/today_candidates.csv --at 15:20:05   # 09-5 Guard 적용
    uv run python -m limitup.execution.paper fills
    uv run python -m limitup.execution.paper sell-open --at 08:55:00   # 다음 날 시가 단일가에 보유분 전량 매도

주문은 모의투자 앱키(KIS_PAPER_*)로만 낸다. 실전 앱키로는 주문 API를 호출하지 않는다.
"""

import argparse
import datetime as dt
import json
import os
import time

import pandas as pd
import requests

from limitup.collectors.kis import KIS
from limitup.execution.guard import Guard
from limitup.paths import PROCESSED

LOG_DIR = PROCESSED / "paper_orders"
AMOUNT = 2_000_000                # 종목당 주문 금액 — 09-5 Guard의 1건 한도와 같게


def account():
    return os.environ["KIS_PAPER_ACCOUNT_NO"], os.environ.get("KIS_PAPER_ACCOUNT_PROD", "01")


def post(kis: KIS, path: str, tr_id: str, body: dict) -> dict:
    kis.limiter.wait()
    headers = {"content-type": "application/json; charset=utf-8",
               "authorization": f"Bearer {kis.access_token()}", "appkey": kis.app_key,
               "appsecret": kis.app_secret, "tr_id": tr_id, "custtype": "P"}
    return requests.post(kis.domain + path, headers=headers, data=json.dumps(body), timeout=15).json()


def balance(kis):
    cano, prod = account()
    b = kis.get("/uapi/domestic-stock/v1/trading/inquire-balance", "VTTC8434R", {
        "CANO": cano, "ACNT_PRDT_CD": prod, "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
        "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""})
    s = b["output2"][0]
    return {"cash": int(s["dnca_tot_amt"]), "eval": int(s["tot_evlu_amt"]), "holdings": len(b["output1"])}


def buy_limit(kis, code: str, price: int, qty: int) -> dict:
    cano, prod = account()
    body = {"CANO": cano, "ACNT_PRDT_CD": prod, "PDNO": code, "ORD_DVSN": "00",
            "ORD_QTY": str(qty), "ORD_UNPR": str(price)}
    r = post(kis, "/uapi/domestic-stock/v1/trading/order-cash", "VTTC0012U", body)
    if r.get("rt_cd") != "0" and "TR" in r.get("msg1", ""):                 # 예전 거래 ID
        r = post(kis, "/uapi/domestic-stock/v1/trading/order-cash", "VTTC0802U", body)
    return r


def holdings(kis) -> list:
    cano, prod = account()
    b = kis.get("/uapi/domestic-stock/v1/trading/inquire-balance", "VTTC8434R", {
        "CANO": cano, "ACNT_PRDT_CD": prod, "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
        "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""})
    return [{"code": h["pdno"], "qty": int(h["hldg_qty"])} for h in b["output1"] if int(h["hldg_qty"]) > 0]


def sell_market(kis, code: str, qty: int) -> dict:
    """시장가 매도. 장 시작 전(08:30~09:00)에 내면 시가 단일가로 체결된다."""
    cano, prod = account()
    body = {"CANO": cano, "ACNT_PRDT_CD": prod, "PDNO": code, "ORD_DVSN": "01",
            "ORD_QTY": str(qty), "ORD_UNPR": "0"}
    r = post(kis, "/uapi/domestic-stock/v1/trading/order-cash", "VTTC0011U", body)
    if r.get("rt_cd") != "0" and "TR" in r.get("msg1", ""):
        r = post(kis, "/uapi/domestic-stock/v1/trading/order-cash", "VTTC0801U", body)
    return r


def fills(kis, day: str) -> pd.DataFrame:
    cano, prod = account()
    params = {"CANO": cano, "ACNT_PRDT_CD": prod, "INQR_STRT_DT": day, "INQR_END_DT": day,
              "SLL_BUY_DVSN_CD": "00", "INQR_DVSN": "00", "PDNO": "", "CCLD_DVSN": "00", "ORD_GNO_BRNO": "",
              "ODNO": "", "INQR_DVSN_3": "00", "INQR_DVSN_1": "", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
    for tr in ("VTTC0081R", "VTTC8001R"):
        try:
            b = kis.get("/uapi/domestic-stock/v1/trading/inquire-daily-ccld", tr, params)
            break
        except Exception as e:                                     # 거래 ID가 바뀐 경우 예전 ID로
            last = e
    else:
        raise last
    cols = {"odno": "order_no", "pdno": "code", "ord_tmd": "order_time", "ord_qty": "qty",
            "ord_unpr": "price", "tot_ccld_qty": "filled_qty", "avg_prvs": "avg_price", "rmn_qty": "remain_qty"}
    df = pd.DataFrame(b.get("output1", []))
    return df[[c for c in cols if c in df]].rename(columns=cols) if not df.empty else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["balance", "buy-close", "fills", "sell-open"])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--at", default="15:20:05", help="이 시각까지 기다렸다가 주문 (장 마감 단일가)")
    ap.add_argument("--amount", type=int, default=AMOUNT, help="종목당 주문 금액")
    a = ap.parse_args()
    kis = KIS("paper")
    today = dt.date.today().strftime("%Y%m%d")
    if a.cmd == "balance":
        print(balance(kis))
    elif a.cmd == "fills":
        print(fills(kis, today).to_string(index=False))
    elif a.cmd == "sell-open":
        target = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(a.at))
        while dt.datetime.now() < target:
            time.sleep(0.5)
        for h in holdings(kis):
            r = sell_market(kis, h["code"], h["qty"])
            print({"time": dt.datetime.now().isoformat(timespec="seconds"), **h, "rt_cd": r.get("rt_cd"),
                   "msg": r.get("msg1", "").strip()}, flush=True)
    else:
        cands = pd.read_csv(a.csv, dtype={"code": str})
        target = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(a.at))
        while dt.datetime.now() < target:
            time.sleep(0.5)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        guard = Guard(kis)                # 모의투자 확인·STOP·주문 수와 금액 한도·중복 방지 (09-5)
        log = []
        for r in cands.itertuples():
            ok, why = guard.allow(r.code, a.amount)
            if not ok:
                print(r.code, "주문 안 함:", why, flush=True)
                continue
            qty = a.amount // r.upper
            res = buy_limit(kis, r.code, int(r.upper), int(qty))
            guard.record(r.code, a.amount, res)
            row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "code": r.code, "price": int(r.upper),
                   "qty": int(qty), "rt_cd": res.get("rt_cd"), "msg": res.get("msg1", "").strip(),
                   "order_no": (res.get("output") or {}).get("ODNO")}
            log.append(row)
            print(row, flush=True)
        (LOG_DIR / f"{today}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
