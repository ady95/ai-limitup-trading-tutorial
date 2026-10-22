"""06-1 첫 상한가 이벤트의 직전 공시 수집 (Open DART)

    uv run python -m limitup.collectors.dart

- 종목코드 → DART 고유번호(corp_code) 연결표: corpCode.xml
- 이벤트마다 T-3일 ~ T일에 접수된 공시 목록: list.json
- 지분 변동 보고처럼 재료가 아닌 정기·형식 공시를 뺀 공시는 본문(document.xml) 앞부분도 받는다

미래정보 누수 방지: 접수일(rcept_dt)이 T일 이하인 공시만 쓴다. T일 장 마감 후 접수된 공시도
판단 시점(장 마감 후 ~ T+1 장 시작 전)에는 볼 수 있으므로 포함한다.
제출인 이름(flr_nm)은 개인일 수 있으므로 저장하지 않는다.
"""

import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from dotenv import load_dotenv

from limitup.collectors.kis import RateLimiter
from limitup.paths import PROCESSED, RAW, ROOT

DART = "https://opendart.fss.or.kr/api"
CORP_FILE = RAW / "dart_corp.parquet"
DISC_FILE = RAW / "dart_disclosures.parquet"
DOC_DIR = RAW / "dart_docs"
LOOKBACK_DAYS = 3
ROUTINE = re.compile(r"임원ㆍ주요주주특정증권등소유상황보고서|주식등의대량보유상황보고서|"
                     r"특정증권등소유상황보고서|자기주식취득결과보고서|증권발행실적보고서")
DOC_CHARS = 2500                        # 본문은 앞부분만 (LLM 입력 길이와 비용 관리)
limiter = RateLimiter(5)                # 초당 5건 — Open DART 이용 한도보다 충분히 낮게


def api_key():
    load_dotenv(ROOT / ".env")
    return os.environ["DART_API_KEY"]


def corp_codes() -> dict:
    if not CORP_FILE.exists():
        raw = requests.get(f"{DART}/corpCode.xml", params={"crtfc_key": api_key()}, timeout=60).content
        z = zipfile.ZipFile(io.BytesIO(raw))
        root = ET.fromstring(z.read(z.namelist()[0]))
        df = pd.DataFrame([{c.tag: (c.text or "").strip() for c in el} for el in root.iter("list")])
        df = df[df.stock_code.str.len() == 6][["corp_code", "stock_code"]]
        df.to_parquet(CORP_FILE, index=False)
    df = pd.read_parquet(CORP_FILE)
    return dict(zip(df.stock_code, df.corp_code))


def list_disclosures(key, corp_code, date):
    limiter.wait()
    bgn = (date - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    j = requests.get(f"{DART}/list.json", timeout=30, params={
        "crtfc_key": key, "corp_code": corp_code, "bgn_de": bgn, "end_de": date.strftime("%Y%m%d"),
        "page_count": 100}).json()
    if j.get("status") not in ("000", "013"):            # 013 = 조회된 데이터 없음
        raise RuntimeError(f"{j.get('status')} {j.get('message')}")
    return [{k: x.get(k) for k in ("rcept_no", "rcept_dt", "report_nm", "rm")} for x in j.get("list", [])]


def document_text(key, rcept_no) -> str:
    """공시 본문 XML에서 태그를 지운 앞부분 텍스트 (파일로 캐시)"""
    path = DOC_DIR / f"{rcept_no}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    limiter.wait()
    raw = requests.get(f"{DART}/document.xml", params={"crtfc_key": key, "rcept_no": rcept_no}, timeout=60).content
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        xml = z.read(sorted(z.namelist())[0])
        for enc in ("utf-8", "euc-kr", "cp949"):
            try:
                text = xml.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()[:DOC_CHARS]
    except zipfile.BadZipFile:
        text = ""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main():
    key, cmap = api_key(), corp_codes()
    ev = pd.read_parquet(PROCESSED / "features.parquet")[["code", "date"]]

    def one(r):
        rows = list_disclosures(key, cmap[r.code], r.date) if r.code in cmap else []
        return [{"code": r.code, "date": r.date, **x} for x in rows]

    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        rows = [x for part in pool.map(one, ev.itertuples()) for x in part]
    disc = pd.DataFrame(rows)
    disc["rcept_dt"] = pd.to_datetime(disc.rcept_dt)
    disc = disc[disc.rcept_dt <= disc.date]                     # 누수 방지 (T일 이하)
    disc["routine"] = disc.report_nm.str.contains(ROUTINE)
    print(f"공시 목록 {len(disc):,}건 ({time.time() - t0:.0f}초), 이벤트 {disc.groupby(['code', 'date']).ngroups:,}/{len(ev):,}건에 공시 있음")

    # 재료 후보(정기·형식 공시 제외) 중 최근 3건의 본문 앞부분
    cand = disc[~disc.routine].sort_values("rcept_no").groupby(["code", "date"]).tail(3)
    with ThreadPoolExecutor(4) as pool:
        texts = list(pool.map(lambda n: document_text(key, n), cand.rcept_no))
    disc = disc.merge(pd.DataFrame({"rcept_no": cand.rcept_no.values, "body": texts}), on="rcept_no", how="left")
    disc.to_parquet(DISC_FILE, index=False)
    print(f"본문 {len(cand):,}건 ({time.time() - t0:.0f}초) → {DISC_FILE}")


if __name__ == "__main__":
    main()
