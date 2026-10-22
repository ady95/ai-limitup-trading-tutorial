"""02-3 분석 모집단 만들기 — 현재 상장 종목 + 2021년 이후 상장폐지 종목

    uv run python -m limitup.collectors.master

현재 상장 종목은 한국투자증권 종목 마스터 파일에서, 상장폐지 종목은 KRX 상장공시시스템(KIND)의
상장폐지 현황을 FinanceDataReader로 가져온다. 둘을 합쳐야 생존편향 없는 모집단이 된다.
"""

import io
import zipfile

import FinanceDataReader as fdr
import pandas as pd
import requests

from limitup.paths import UNIVERSE

URL = "https://new.real.download.dws.co.kr/common/master/{}_code.mst.zip"
SINCE = "2021-01-01"   # 이 날짜 이후 폐지된 종목까지 포함

# 한 줄 = 앞부분(단축코드 9 + 표준코드 12 + 종목명) + 뒷부분(고정폭 속성).
# 뒷부분 칸 너비는 한국투자증권 공식 저장소 stocks_info/kis_*_code_mst.py 기준.
KOSPI_WIDTHS = [2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8,
                15, 21, 2, 7, 1, 1, 1, 1, 1, 9, 9, 9, 5, 9, 8, 9, 3, 1, 1, 1]
KOSDAQ_WIDTHS = [2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 1, 3, 12, 12, 8, 15, 21, 2, 7,
                 1, 1, 1, 1, 9, 9, 9, 5, 9, 8, 9, 3, 1, 1, 1]
# 시장별로 필요한 칸의 번호 (공식 저장소의 열 이름 순서 기준)
COLS = {
    "kospi": {"spac": 19, "halted": 34, "managed": 36, "listed": 49, "preferred": 54},
    "kosdaq": {"spac": 14, "halted": 29, "managed": 31, "listed": 44, "preferred": 49},
}
WIDTHS = {"kospi": KOSPI_WIDTHS, "kosdaq": KOSDAQ_WIDTHS}


def download(market):
    raw = requests.get(URL.format(market), timeout=60).content
    return zipfile.ZipFile(io.BytesIO(raw)).read(f"{market}_code.mst").decode("cp949")


def parse(market, text):
    widths, cols = WIDTHS[market], COLS[market]
    tail = sum(widths)                     # 뒷부분 길이 (줄바꿈 제외)
    starts = [sum(widths[:i]) for i in range(len(widths))]

    def field(attrs, name):
        i = cols[name]
        return attrs[starts[i]:starts[i] + widths[i]].strip()

    rows = []
    for line in text.splitlines():
        head, attrs = line[:-tail], line[-tail:]
        rows.append({
            "code": head[0:9].strip(),
            "name": head[21:].strip(),
            "market": market.upper(),
            "group_code": attrs[0:2],
            "is_spac": field(attrs, "spac") == "Y",
            "is_halted": field(attrs, "halted") == "Y",
            "is_managed": field(attrs, "managed") == "Y",
            "preferred": field(attrs, "preferred"),
            "listed": field(attrs, "listed"),
        })
    return pd.DataFrame(rows)


def listed_stocks():
    """현재 상장된 KOSPI·KOSDAQ 보통주 (주권 ST, 스팩·우선주 제외)"""
    df = pd.concat([parse(m, download(m)) for m in WIDTHS], ignore_index=True)
    keep = (df.group_code == "ST") & ~df.is_spac & (df.preferred == "0") \
        & df.code.str.fullmatch(r"\d{6}")
    out = df[keep].copy()
    out["listed"] = pd.to_datetime(out.listed, format="%Y%m%d", errors="coerce")
    out["delisted"] = pd.NaT
    return out[["code", "name", "market", "listed", "delisted", "is_halted", "is_managed"]]


def delisted_stocks():
    """SINCE 이후 상장폐지된 KOSPI·KOSDAQ 보통주 (스팩·투자회사 제외)"""
    d = fdr.StockListing("KRX-DELISTING")
    d["DelistingDate"] = pd.to_datetime(d.DelistingDate)
    keep = (d.Kind == "보통주") & d.Market.isin(["KOSPI", "KOSDAQ"]) \
        & d.Symbol.str.fullmatch(r"\d{6}") & (d.DelistingDate >= SINCE) \
        & ~d.Name.str.contains("스팩|기업인수목적") \
        & ~d.Reason.fillna("").str.contains("해산 사유|증권투자회사법")
    out = d[keep].rename(columns={"Symbol": "code", "Name": "name", "Market": "market",
                                  "ListingDate": "listed", "DelistingDate": "delisted"})
    out["listed"] = pd.to_datetime(out.listed)
    out["is_halted"] = False
    out["is_managed"] = False
    out["reason"] = d.Reason
    return out[["code", "name", "market", "listed", "delisted", "is_halted", "is_managed", "reason"]]


def main():
    listed, delisted = listed_stocks(), delisted_stocks()
    # 같은 코드가 폐지 후 재상장된 경우 현재 상장 쪽을 남긴다
    delisted = delisted[~delisted.code.isin(listed.code)].drop_duplicates("code", keep="last")
    universe = pd.concat([listed, delisted], ignore_index=True).sort_values("code")
    UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(UNIVERSE, index=False)

    summary = universe.assign(status=universe.delisted.isna().map({True: "상장", False: "폐지"}))
    print(summary.groupby(["market", "status"]).size().unstack().to_string())
    print(f"합계 {len(universe):,}종목 → {UNIVERSE}")


if __name__ == "__main__":
    main()
