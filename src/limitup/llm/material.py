"""06-2 LLM으로 첫 상한가의 재료(공시) 분석 — 구조화된 JSON으로

    uv run python -m limitup.llm.material --sample 50   # 표본으로 품질·시간 확인
    uv run python -m limitup.llm.material               # 전체 (결과는 이벤트별 파일로 캐시)

원칙
- "내일 오를까?"를 묻지 않는다. 공시가 기업 가치에 주는 영향을 정해진 항목으로 분류하게 한다
- 회사명·종목코드를 '대상회사'로 가린다. 모델이 학습 때 본 그 회사의 이후 주가를 쓰지 못하게
- 근거로 쓴 공시의 접수번호를 함께 내게 해 검증할 수 있게 한다
- API는 OpenAI 호환 Responses API (엄격한 JSON 스키마 출력). 모델은 .env 의 OPENAI_MODEL
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from limitup.collectors.dart import DISC_FILE
from limitup.paths import PROCESSED, ROOT, UNIVERSE

OUT_DIR = PROCESSED / "llm"
EVENT_TYPES = ["supply_contract", "earnings", "capital_raise", "control_change", "mna",
               "tech_pharma", "policy_theme", "shareholder_return", "legal_risk", "trading_alert",
               "ownership_filing", "routine", "other"]
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["event_type", "material_category", "financial_impact", "impact_duration",
                 "uncertainty", "evidence", "confidence", "summary"],
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "material_category": {"type": "string",
                              "enum": ["fundamental", "financing", "governance", "speculative", "negative", "none"]},
        "financial_impact": {"type": "string",
                             "enum": ["significant", "moderate", "minor", "none", "negative"]},
        "impact_duration": {"type": "string", "enum": ["one_day", "short_term", "medium_term", "long_term"]},
        "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
    },
}
INSTRUCTIONS = """당신은 한국 주식시장의 공시를 분석하는 애널리스트입니다.
대상회사의 주가가 오늘 상한가로 마감했습니다. 아래 공시 목록과 본문 발췌만을 근거로, 이 상한가의 재료가
기업 가치에 어떤 영향을 주는지 정해진 항목으로 분류하세요.

규칙
- 제공된 공시 외의 정보(회사에 대한 사전 지식, 이후의 주가 움직임 등)는 절대 쓰지 마세요.
- 공시가 재료가 아니라 형식적인 보고(지분 변동 보고 등)뿐이면 event_type 을 ownership_filing 또는 routine 으로 두세요.
- evidence 에는 판단 근거가 된 공시의 접수번호(rcept_no)만 넣으세요.
- confidence 는 분류에 대한 확신(0~1)이며, 주가 상승 확률이 아닙니다.
- summary 는 한국어 60자 이내로 재료를 요약하세요.

항목 설명
- event_type: supply_contract(공급계약), earnings(실적), capital_raise(유상증자·전환사채 등 자금조달),
  control_change(최대주주 변경), mna(합병·분할·인수), tech_pharma(기술·신약·특허), policy_theme(정책·테마),
  shareholder_return(배당·자사주), legal_risk(소송·감사·불성실공시), trading_alert(시장경보·투자주의 등),
  ownership_filing(지분 보고), routine(정기·형식 공시), other
- financial_impact: 매출·이익·자본에 주는 영향의 크기 (significant, moderate, minor, none, negative)
- impact_duration: 영향이 지속될 기간 (one_day, short_term, medium_term, long_term)
- uncertainty: 계약 취소·조건 미확정 등 실현 불확실성 (low, medium, high)"""


def client():
    load_dotenv(ROOT / ".env")
    return OpenAI(), os.environ["OPENAI_MODEL"]       # OPENAI_BASE_URL·OPENAI_API_KEY 는 .env 에서


def mask(text: str, name: str, code: str) -> str:
    for s in (name, code):
        if s:
            text = text.replace(s, "대상회사")
    return text


def build_prompt(disc: pd.DataFrame, name: str, code: str) -> str:
    lines = ["[공시 목록] (접수일, 접수번호, 제목)"]
    for r in disc.sort_values("rcept_no").itertuples():
        lines.append(f"- {r.rcept_dt:%Y-%m-%d} {r.rcept_no} {mask(r.report_nm.strip(), name, code)}")
    for r in disc[disc.body.fillna("") != ""].itertuples():
        lines.append(f"\n[본문 발췌 {r.rcept_no}]\n{mask(r.body, name, code)}")
    return "\n".join(lines)


def analyze(cli, model, prompt: str) -> dict:
    r = cli.responses.create(model=model, instructions=INSTRUCTIONS, input=prompt,
                             text={"format": {"type": "json_schema", "name": "material",
                                              "strict": True, "schema": SCHEMA}})
    out = json.loads(r.output_text)
    out["confidence"] = min(max(float(out["confidence"]), 0.0), 1.0)
    return out


def run(events: pd.DataFrame, workers=6):
    cli, model = client()
    disc = pd.read_parquet(DISC_FILE)
    names = pd.read_parquet(UNIVERSE).set_index("code").name.to_dict()
    groups = {k: g for k, g in disc.groupby(["code", "date"])}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def one(r):
        path = OUT_DIR / f"{r.code}_{r.date:%Y%m%d}.json"
        if path.exists():
            return "cached"
        g = groups.get((r.code, r.date))
        if g is None:
            res = {"has_disc": 0}
        elif g.routine.all():
            res = {"has_disc": 1, "event_type": "routine_only", "n_disc": len(g)}
        else:
            t = time.time()
            res = {"has_disc": 1, "n_disc": len(g), "n_material": int((~g.routine).sum()),
                   **analyze(cli, model, build_prompt(g, names.get(r.code, ""), r.code)),
                   "model": model, "seconds": round(time.time() - t, 1)}
        path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return "llm" if "seconds" in res else "skip"

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        stat = pd.Series(list(pool.map(one, events.itertuples()))).value_counts()
    print(stat.to_string(), f"\n{len(events):,}건 처리 {time.time() - t0:.0f}초")


def load_results() -> pd.DataFrame:
    rows = []
    for p in OUT_DIR.glob("*.json"):
        code, day = p.stem.split("_")
        rows.append({"code": code, "date": pd.Timestamp(day), **json.loads(p.read_text(encoding="utf-8"))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ev = pd.read_parquet(PROCESSED / "features.parquet")[["code", "date", "split"]]
    if args.sample:
        ev = ev.sample(args.sample, random_state=7)
    run(ev, args.workers)


if __name__ == "__main__":
    main()
