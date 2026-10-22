"""04-2 정규장 1분봉 수집 — 이벤트 종목의 T일·T+1일

    uv run python -m limitup.collectors.minute              # 03장 이벤트(세 집단)의 T, T+1
    uv run python -m limitup.collectors.minute 005930:20261021

한국투자증권 일별 분봉 API(FHKST03010230)는 실측 결과 **최근 약 1년**(2026-10-22 기준 2025-10-14 이후)만
돌려준다. 오래된 날부터 매일 조회할 수 없게 되므로, 필요한 분봉은 미리 받아 data/raw/minute 에 쌓아 둔다.
"""

import argparse
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from limitup.collectors.kis import KIS, KISError
from limitup.paths import PROCESSED, RAW

MINUTE_DIR = RAW / "minute"
PATH = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
TR_ID = "FHKST03010230"       # 주식 일별 분봉 조회, 한 번에 최대 120행 (최근 시각부터 과거 방향)
COLUMNS = {"stck_cntg_hour": "time", "stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low",
           "stck_prpr": "close", "cntg_vol": "volume", "acml_tr_pbmn": "cum_value"}


def fetch_day(kis, code, day):
    """하루 정규장(09:00~15:30) 분봉. 받은 행의 가장 이른 시각 1분 전부터 다시 요청한다."""
    rows, hour = {}, "153000"
    while hour >= "090000":
        body = kis.get(PATH, TR_ID, {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_HOUR_1": hour,
            "FID_INPUT_DATE_1": day, "FID_PW_DATA_INCU_YN": "N", "FID_FAKE_TICK_INCU_YN": ""})
        page = [r for r in body.get("output2", []) if r.get("stck_bsop_date") == day
                and "090000" <= r.get("stck_cntg_hour", "") <= "153000"]
        new = [r for r in page if r["stck_cntg_hour"] not in rows]
        if not new:
            break
        rows.update({r["stck_cntg_hour"]: r for r in new})
        earliest = dt.datetime.strptime(min(r["stck_cntg_hour"] for r in new), "%H%M%S")
        hour = (earliest - dt.timedelta(minutes=1)).strftime("%H%M%S")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows.values())[list(COLUMNS)].rename(columns=COLUMNS)
    df[["open", "high", "low", "close", "volume", "cum_value"]] = \
        df[["open", "high", "low", "close", "volume", "cum_value"]].astype("int64")
    df.insert(0, "date", pd.Timestamp(day))
    df.insert(0, "code", code)
    return df.sort_values("time", ignore_index=True)


def collect(kis, code, day):
    path = MINUTE_DIR / day / f"{code}.parquet"
    if path.exists():
        return -1                                  # 이미 받음
    df = fetch_day(kis, code, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)               # 빈 결과도 저장해 다시 묻지 않는다
    return len(df)


def event_targets(since="2025-10-14"):
    """03장 이벤트(마감·터치·대조군)의 T일과 T+1일"""
    ev = pd.read_parquet(PROCESSED / "events_with_control.parquet")
    ev = ev[ev.date >= since]
    pairs = set(zip(ev.code, ev.date.dt.strftime("%Y%m%d")))
    t1 = ev.dropna(subset=["t1_date"])
    pairs |= set(zip(t1.code, t1.t1_date.dt.strftime("%Y%m%d")))
    return sorted(pairs, key=lambda p: (p[1], p[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="종목코드:YYYYMMDD (생략하면 03장 이벤트)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pairs = [tuple(t.split(":")) for t in args.targets] or event_targets()

    kis = KIS("real")
    t0, done, empty, skipped, failed = time.time(), 0, 0, 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(collect, kis, c, d): (c, d) for c, d in pairs}
        for fut in as_completed(futures):
            done += 1
            try:
                n = fut.result()
                skipped += n == -1
                empty += n == 0
            except KISError as e:
                failed.append(f"{futures[fut]}\t{e}")
            if done % 500 == 0 or done == len(futures):
                el = time.time() - t0
                print(f"{done:>5}/{len(futures)}  빈 날 {empty}  기존 {skipped}  경과 {el / 60:.1f}분  "
                      f"남은 시간 약 {el / done * (len(futures) - done) / 60:.1f}분", flush=True)
    if failed:
        (RAW / "minute_failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"실패 {len(failed)}건 → {RAW / 'minute_failed.txt'}")


if __name__ == "__main__":
    main()
