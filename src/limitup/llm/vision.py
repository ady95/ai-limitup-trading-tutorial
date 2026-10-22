"""06-5 VLM으로 AI 분석용 차트 읽기 — 상한가 형성 패턴을 구조화된 JSON으로

    uv run python -m limitup.llm.vision --sample 30
    uv run python -m limitup.llm.vision

이미지는 base64로 Responses API에 넣는다. 결과는 이벤트별 파일로 캐시한다.
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from limitup.llm.chart import CHART_DIR
from limitup.llm.material import client
from limitup.paths import PROCESSED

OUT_DIR = PROCESSED / "vlm"
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["pattern", "first_touch_phase", "lock_strength", "selling_pressure",
                 "volume_profile", "description"],
    "properties": {
        "pattern": {"type": "string", "enum": ["early_lock", "gradual", "repeated_break", "late"]},
        "first_touch_phase": {"type": "string", "enum": ["first_hour", "morning", "afternoon", "last_hour"]},
        "lock_strength": {"type": "integer", "minimum": 1, "maximum": 5},
        "selling_pressure": {"type": "integer", "minimum": 1, "maximum": 5},
        "volume_profile": {"type": "string", "enum": ["front_loaded", "even", "back_loaded"]},
        "description": {"type": "string"},
    },
}
INSTRUCTIONS = """당신은 주식 차트 분석가입니다. 그림은 어떤 종목의 하루 1분봉 차트입니다.
위 칸: 검은 선 = 기준가 대비 등락률(가로축은 장 시작부터 장 마감까지), 빨간 점선 = 상한가, 회색 선 = 기준가(0%).
아래 칸: 분당 거래량(그날 최대값 기준 상대 크기).
이 종목은 그날 상한가로 마감했습니다. 그림에 보이는 것만으로 다음 항목을 판단하세요.
- pattern: early_lock(장 초반 상한가 도달 후 거의 이탈 없음), gradual(점진적으로 상승해 도달),
  repeated_break(상한가 도달과 이탈을 여러 번 반복), late(장 후반에 처음 도달)
- first_touch_phase: 처음 상한가에 닿은 시간대
- lock_strength: 상한가에 붙어 있는 견고함 1(약함)~5(강함)
- selling_pressure: 상한가 부근에서 보이는 매도 압력 1(약함)~5(강함)
- volume_profile: 거래량이 하루 중 어디에 몰렸는가
- description: 한국어 40자 이내 요약"""


def analyze(cli, model, png_path) -> dict:
    b64 = base64.b64encode(png_path.read_bytes()).decode()
    r = cli.responses.create(
        model=model, instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": "이 차트를 분석하세요."},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}]}],
        text={"format": {"type": "json_schema", "name": "chart", "strict": True, "schema": SCHEMA}})
    return json.loads(r.output_text)


def run(events, workers=6):
    cli, model = client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def one(r):
        out = OUT_DIR / f"{r.code}_{r.date:%Y%m%d}.json"
        png = CHART_DIR / f"{r.code}_{r.date:%Y%m%d}.png"
        if out.exists() or not png.exists():
            return "skip"
        t = time.time()
        res = {**analyze(cli, model, png), "model": model, "seconds": round(time.time() - t, 1)}
        out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return "vlm"

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        stat = pd.Series(list(pool.map(one, events.itertuples()))).value_counts()
    print(stat.to_string(), f"\n{len(events):,}건 처리 {time.time() - t0:.0f}초")


def load_results() -> pd.DataFrame:
    rows = [{"code": p.stem.split("_")[0], "date": pd.Timestamp(p.stem.split("_")[1]),
             **json.loads(p.read_text(encoding="utf-8"))} for p in OUT_DIR.glob("*.json")]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ev = pd.read_parquet(PROCESSED / "features.parquet")
    ev = ev[ev.has_intraday][["code", "date"]]
    if args.sample:
        ev = ev.sample(args.sample, random_state=7)
    run(ev, args.workers)


if __name__ == "__main__":
    main()
