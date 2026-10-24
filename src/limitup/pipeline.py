"""10-1 데이터 수집부터 예측까지 — 장 마감 후 일일 배치

    uv run python -m limitup.pipeline                 # 매일: 일봉·지수 → 이벤트 → 분봉 → 피처 → 점수
    uv run python -m limitup.pipeline --weekly        # 주 1회: + 종목 마스터(모집단·업종) 갱신, 공시·LLM 분석
    uv run python -m limitup.pipeline --latest        # 분석 기준일을 책의 2026-10-21 대신 최신 확정일로

각 단계는 2~7장의 모듈을 그대로 부른다. 단계마다 걸린 시간을 기록하고, 실패하면 거기서 멈춘다.
수집 당일 일봉은 애프터마켓 때문에 미확정이므로(02-5) 저녁 8시 이후, 가능하면 다음 날 새벽에 돌린다.
"""

import argparse
import datetime as dt
import os
import sys
import time

STEPS = [
    ("master", "모집단·업종 갱신 (02-3)", "limitup.collectors.master", True),
    ("daily", "일봉 증분 수집 (02-3)", "limitup.collectors.daily", False),
    ("index", "지수 일봉 (04-3)", "limitup.collectors.index", False),
    ("events", "첫 상한가 탐지·통계 (03장)", "limitup.events.stats", False),
    ("minute", "새 이벤트의 1분봉 (04-2, 최근 1년만 조회 가능)", "limitup.collectors.minute", False),
    ("dart", "공시 수집 (06-1)", "limitup.collectors.dart", True),
    ("llm", "LLM 재료 분석 (06-2)", "limitup.llm.material", True),
    ("features", "피처·라벨 (04장)", "limitup.features.build", False),
    ("scores", "워크포워드 점수 (07-1)", "limitup.strategies.scores", False),
]


def run_module(module: str):
    import importlib
    import runpy
    sys.argv = [module]
    importlib.invalidate_caches()
    runpy.run_module(module, run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true", help="주 1회 단계(모집단·공시·LLM)까지")
    ap.add_argument("--only", nargs="*", help="이 단계만 실행 (예: --only daily events)")
    ap.add_argument("--latest", action="store_true", help="분석 기준일 = 어제 (오늘 일봉은 미확정일 수 있음)")
    a = ap.parse_args()
    if a.latest:
        os.environ["LIMITUP_END"] = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    print("분석 기준일:", os.environ.get("LIMITUP_END", "2026-10-21 (책 기준)"))
    log = []
    for key, name, module, weekly in STEPS:
        if a.only and key not in a.only:
            continue
        if weekly and not (a.weekly or a.only):
            continue
        t0 = time.time()
        print(f"\n===== [{key}] {name}", flush=True)
        try:
            run_module(module)
        except SystemExit:
            pass
        except Exception as e:                                     # 실패하면 뒤 단계는 돌리지 않는다
            log.append((key, name, time.time() - t0, f"실패: {type(e).__name__}: {e}"))
            break
        log.append((key, name, time.time() - t0, "완료"))
    print("\n===== 요약")
    for key, name, sec, status in log:
        print(f"  {key:9s} {sec / 60:5.1f}분  {status}  — {name}")


if __name__ == "__main__":
    main()
