"""04-3 KOSPI·KOSDAQ 지수 일봉 수집

    uv run python -m limitup.collectors.index

결과: data/raw/index.parquet (date, index, open, high, low, close, volume, value)
"""

import datetime as dt

import pandas as pd

from limitup.collectors.daily import START
from limitup.collectors.kis import KIS
from limitup.paths import RAW

INDEX_FILE = RAW / "index.parquet"
INDICES = {"0001": "KOSPI", "1001": "KOSDAQ"}
PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
TR_ID = "FHKUP03500100"       # 국내주식 업종기간별시세, 한 번에 최대 50행
COLUMNS = {"stck_bsop_date": "date", "bstp_nmix_oprc": "open", "bstp_nmix_hgpr": "high",
           "bstp_nmix_lwpr": "low", "bstp_nmix_prpr": "close", "acml_vol": "volume",
           "acml_tr_pbmn": "value"}


def fetch_index(kis, code, start, end):
    rows, d2 = [], end
    while d2 >= start:
        body = kis.get(PATH, TR_ID, {
            "FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start.strftime("%Y%m%d"), "FID_INPUT_DATE_2": d2.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D"})
        page = [r for r in body.get("output2", []) if r.get("stck_bsop_date")]
        if not page:
            break
        rows += page
        oldest = dt.datetime.strptime(page[-1]["stck_bsop_date"], "%Y%m%d").date()
        if len(page) < 50:
            break
        d2 = oldest - dt.timedelta(days=1)
    df = pd.DataFrame(rows)[list(COLUMNS)].rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df[["volume", "value"]] = df[["volume", "value"]].astype("int64")
    df.insert(1, "index", INDICES[code])
    return df.drop_duplicates("date").sort_values("date", ignore_index=True)


def main():
    kis = KIS("real")
    end = dt.date.today()
    df = pd.concat([fetch_index(kis, c, START, end) for c in INDICES], ignore_index=True)
    df.to_parquet(INDEX_FILE, index=False)
    print(df.groupby("index").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max")))


if __name__ == "__main__":
    main()
