"""02-3 전 종목 일봉 수집 — 원주가 기준, 종목별 Parquet 저장

    uv run python -m limitup.collectors.daily            # 모집단 전체
    uv run python -m limitup.collectors.daily 005930     # 일부 종목만

이미 받은 종목은 마지막 날짜 근처부터만 받는다(증분 수집). 중간에 끊겨도 다시 실행하면 이어진다.

주의: 장 마감 후 당일 일봉에는 KRX 애프터마켓(16:00~20:00) 체결이 반영되어 종가가 정규장 종가와
다를 수 있다. 지난 날짜의 일봉은 정규장 기준으로 확정되므로, 증분 수집 때 최근 REFRESH_DAYS일을
다시 받아 덮어쓴다 (02-5 참조).
"""

import argparse
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from limitup.collectors.kis import KIS, KISError
from limitup.paths import DAILY_DIR, RAW, UNIVERSE

START = dt.date(2020, 9, 1)   # 2021년 초 첫 상한가 판정에 직전 60거래일이 필요하다
REFRESH_DAYS = 7              # 증분 수집 때 다시 받아 덮어쓸 최근 기간 (미확정 당일 일봉 대비)
PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
TR_ID = "FHKST03010100"       # 국내주식 기간별 시세(일/주/월/년), 한 번에 최대 100행

COLUMNS = {
    "stck_bsop_date": "date", "stck_oprc": "open", "stck_hgpr": "high", "stck_lwpr": "low",
    "stck_clpr": "close", "acml_vol": "volume", "acml_tr_pbmn": "value",
    "prdy_vrss": "change", "prdy_vrss_sign": "sign",
    "flng_cls_code": "flng_cls_code", "mod_yn": "mod_yn",
}
INT_COLS = ["open", "high", "low", "close", "volume", "value", "change"]


def fetch_range(kis, code, start, end):
    """start~end 일봉을 100행씩 끊어 과거 방향으로 받는다."""
    rows, d2 = [], end
    while d2 >= start:
        body = kis.get(PATH, TR_ID, {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start.strftime("%Y%m%d"), "FID_INPUT_DATE_2": d2.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",          # 1 = 원주가, 0 = 수정주가
        })
        page = [r for r in body.get("output2", []) if r.get("stck_bsop_date")]
        if not page:
            break
        rows += page
        oldest = dt.datetime.strptime(page[-1]["stck_bsop_date"], "%Y%m%d").date()
        if len(page) < 100:
            break
        d2 = oldest - dt.timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["code", *COLUMNS.values(), "base"])
    df = pd.DataFrame(rows)[list(COLUMNS)].rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
    df[INT_COLS] = df[INT_COLS].astype("int64")
    df["base"] = df.close - df.change          # 기준가격 = 종가 - 전일대비
    df.insert(0, "code", code)
    return df.drop_duplicates("date").sort_values("date", ignore_index=True)


def collect(kis, code, end):
    """한 종목을 증분 수집해 저장하고, 새로 받은 행 수를 돌려준다."""
    path = DAILY_DIR / f"{code}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    if old is None or old.empty:
        start = START
    elif end < dt.date.today() - dt.timedelta(days=REFRESH_DAYS):
        return 0                               # 이미 끝난 폐지 종목은 다시 받지 않는다
    else:
        start = old.date.max().date() - dt.timedelta(days=REFRESH_DAYS)
    new = fetch_range(kis, code, start, end)
    if old is None:
        new.to_parquet(path, index=False)
        return len(new)
    if new.empty:
        return 0
    merged = pd.concat([old, new]).drop_duplicates("date", keep="last").sort_values("date")
    merged.to_parquet(path, index=False)        # 최근 행은 새로 받은 값으로 덮어쓴다
    return len(merged) - len(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", help="종목코드 (생략하면 모집단 전체)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    uni = pd.read_parquet(UNIVERSE)
    if args.codes:
        uni = uni[uni.code.isin(args.codes)]
    today = dt.date.today()
    # 폐지 종목은 폐지일 전날까지만 받는다
    ends = {r.code: (r.delisted.date() - dt.timedelta(days=1) if pd.notna(r.delisted) else today)
            for r in uni.itertuples()}

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    kis = KIS("real")
    t0, done, added, failed = time.time(), 0, 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(collect, kis, c, e): c for c, e in ends.items()}
        for fut in as_completed(futures):
            done += 1
            try:
                added += fut.result()
            except KISError as e:
                failed.append(f"{futures[fut]}\t{e}")
            if done % 100 == 0 or done == len(futures):
                el = time.time() - t0
                print(f"{done:>5}/{len(futures)}  새 행 {added:,}  경과 {el / 60:.1f}분  "
                      f"남은 시간 약 {el / done * (len(futures) - done) / 60:.1f}분", flush=True)
    if failed:
        (RAW / "daily_failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"실패 {len(failed)}종목 → {RAW / 'daily_failed.txt'}")


if __name__ == "__main__":
    main()
