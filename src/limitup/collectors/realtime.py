"""09-1 실시간 체결가·호가 수신 (WebSocket) — 장중에만 데이터가 온다

    uv run python -m limitup.collectors.realtime 005930 000660 --seconds 60

체결가(H0STCNT0)와 호가(H0STASP0)를 함께 구독해 data/raw/realtime/<날짜>/ 에 저장한다.
호가의 매수 1호가 잔량은 상한가에서 '대기열의 크기'가 된다 — 과거 데이터로는 얻을 수 없었던 정보(08-3).
"""

import argparse
import asyncio
import datetime as dt
import json
from collections import Counter

import pandas as pd
import websockets

from limitup.collectors.kis import KIS
from limitup.paths import RAW

WS_URLS = {"real": "ws://ops.koreainvestment.com:21000", "paper": "ws://ops.koreainvestment.com:31000"}
OUT_DIR = RAW / "realtime"

# 필드 순서: 한국투자증권 공식 예제 (체결가 ccnl_krx, 호가 asking_price_krx)
TRADE_COLS = [
    "code", "time", "price", "sign", "change", "change_rate", "wavg", "open", "high", "low",
    "ask1", "bid1", "volume", "acc_volume", "acc_value", "sell_cnt", "buy_cnt", "net_cnt", "strength",
    "sell_vol", "buy_vol", "side", "buy_rate", "vol_rate", "open_time", "open_sign", "open_diff",
    "high_time", "high_sign", "high_diff", "low_time", "low_sign", "low_diff", "date", "new_mkop",
    "halt", "ask_qty1", "bid_qty1", "total_ask_qty", "total_bid_qty", "turnover", "prev_vol",
    "prev_vol_rate", "hour_cls", "mrkt_trtm", "vi_price"]
QUOTE_COLS = (["code", "time", "hour_cls"] + [f"ask{i}" for i in range(1, 11)] + [f"bid{i}" for i in range(1, 11)]
              + [f"ask_qty{i}" for i in range(1, 11)] + [f"bid_qty{i}" for i in range(1, 11)]
              + ["total_ask_qty", "total_bid_qty", "ovtm_total_ask", "ovtm_total_bid", "exp_price", "exp_qty",
                 "exp_vol", "exp_change", "exp_sign", "exp_rate", "acc_volume", "total_ask_chg", "total_bid_chg",
                 "ovtm_ask_chg", "ovtm_bid_chg", "stck_deal_cls"])
TRS = {"H0STCNT0": TRADE_COLS, "H0STASP0": QUOTE_COLS}


def subscribe(approval_key, tr_id, code, tr_type="1"):
    return json.dumps({"header": {"approval_key": approval_key, "custtype": "P", "tr_type": tr_type,
                                  "content-type": "utf-8"},
                       "body": {"input": {"tr_id": tr_id, "tr_key": code}}})


def parse(raw):
    """'0|TR_ID|건수|필드^필드^...' → (tr_id, dict 목록)"""
    _, tr_id, count, payload = raw.split("|", 3)
    cols, values = TRS[tr_id], payload.split("^")
    n = len(values) // int(count)
    return tr_id, [dict(zip(cols, values[i * n:(i + 1) * n])) for i in range(int(count))]


async def run(codes, seconds, env="real"):
    kis = KIS(env)
    key = kis.approval_key()
    rows = {tr: [] for tr in TRS}
    stats = Counter()
    async with websockets.connect(WS_URLS[env], open_timeout=10) as ws:
        for code in codes:                      # 한 세션 구독 한도(약 41건) 안에서, 간격을 두고 등록
            for tr in TRS:
                await ws.send(subscribe(key, tr, code))
                await asyncio.sleep(0.2)
        end = asyncio.get_running_loop().time() + seconds
        while (left := end - asyncio.get_running_loop().time()) > 0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=left)
            except asyncio.TimeoutError:
                break
            if raw[0] in "01":
                tr, recs = parse(raw)
                now = dt.datetime.now().isoformat(timespec="milliseconds")
                rows[tr] += [{**r, "recv": now} for r in recs]
                stats[tr] += len(recs)
            else:
                msg = json.loads(raw)
                if msg["header"]["tr_id"] == "PINGPONG":
                    await ws.pong(raw)
                    stats["pingpong"] += 1
                else:
                    stats[f"응답 {msg.get('body', {}).get('msg1', '').strip()}"] += 1
    day = dt.date.today().strftime("%Y%m%d")
    (OUT_DIR / day).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%H%M%S")
    for tr, recs in rows.items():
        if recs:
            pd.DataFrame(recs).to_parquet(OUT_DIR / day / f"{tr}_{stamp}.parquet", index=False)
    return stats, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+")
    ap.add_argument("--seconds", type=int, default=60)
    a = ap.parse_args()
    stats, rows = asyncio.run(run(a.codes, a.seconds))
    print(dict(stats))
    q = rows["H0STASP0"]
    if q:
        last = pd.DataFrame(q).groupby("code").tail(1)
        print(last[["code", "time", "hour_cls", "ask1", "bid1", "ask_qty1", "bid_qty1", "total_ask_qty", "total_bid_qty"]].to_string(index=False))


if __name__ == "__main__":
    main()
