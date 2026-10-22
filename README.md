# AI로 첫 상한가 종목의 연속 상승 예측과 자동매매 따라하기 — 예제 코드

위키독스 책 「[AI로 첫 상한가 종목의 연속 상승 예측과 자동매매 따라하기](https://wikidocs.net/book/21798)」의 실습 코드 저장소입니다.

> 이 저장소는 교육·연구 목적의 예제 코드이며 투자 권유가 아닙니다. 주문 실습은 모의투자 환경에서만 진행합니다.

## 빠른 시작

```bash
git clone https://github.com/ady95/ai-limitup-trading-tutorial.git
cd ai-limitup-trading-tutorial
uv sync                      # Python 3.11+ 가상환경과 패키지 설치
cp .env.example .env         # 앱키 등을 채운다 (.env 는 커밋 금지)
uv run pytest -q             # 상한가 계산 테스트
```

## 장별 실행 순서

| 장 | 명령 | 결과물 |
|---|---|---|
| 02-2 | `uv run python -m limitup.collectors.quote 005930` | 현재가·기준가·상한가 조회 |
| 02-3 | `uv run python -m limitup.collectors.master` | `data/raw/universe.parquet` (상장 + 2021년 이후 폐지 종목) |
| 02-3 | `uv run python -m limitup.collectors.daily` | `data/raw/daily/<종목코드>.parquet` (2020-09-01~, 원주가) |
| 02-4 | `uv run python -m limitup.storage` | `data/limitup.duckdb` (뷰 `daily`, `universe`, `market_daily`) |
| 02-5 | `uv run python -m limitup.collectors.check` | 정합성 점검 리포트 |
| 02-5 | `from limitup.collectors.adjust import adjust` | 기준가로 만든 수정주가 |
| 03-1 | `from limitup.events.price_limit import upper_limit, lower_limit, add_limits` | 거래소 방식 상한가·하한가 (거래소 표시와 전 종목 일치) |
| 03-2 | `uv run python -m limitup.events.detect --lookback 20 40 60 120` | `data/processed/events.parquet` (첫 상한가 이벤트) |
| 03-3~5 | `uv run python -m limitup.events.stats` | 다음 날 통계, H1 검증, `data/processed/report_ch03.html` |
| 04-2 | `uv run python -m limitup.collectors.minute` | `data/raw/minute/<날짜>/<종목코드>.parquet` (이벤트일 정규장 1분봉, 최근 약 1년만 조회 가능) |
| 04-3 | `uv run python -m limitup.collectors.index` | `data/raw/index.parquet` (KOSPI·KOSDAQ 지수 일봉) |
| 04-3 | `uv run python -m limitup.collectors.master` | `data/raw/industry.parquet` (업종·주요 제품) |
| 04-1~5 | `uv run python -m limitup.features.build` | `data/processed/features.parquet` (피처 26개, 라벨, 기간 분할) |
| 04-4 | `uv run python -m limitup.features.build --leak 300` | 미래정보 누수 검사 |
| 04-1~3 | `uv run python -m limitup.features.build --explore` | 학습 기간 안에서 피처 구간별 통계 |
| 05-1 | `uv run python -m limitup.models.baseline` | 규칙 기반 기준 모델 M0 |
| 05-2~3 | `uv run python -m limitup.models.lgbm` (`--walk`, `--intraday`) | LightGBM M1, 워크포워드, 장중 피처 추가 효과 |
| 05-4 | `uv run python -m limitup.models.regress` | 시가 갭·장중 수익 회귀, 하방 위험 분위수 |
| 05-5 | `uv run python -m limitup.models.explain` | SHAP 해석 |
| 05-6 | `uv run python -m limitup.models.calibrate` | 신뢰도표, Brier·ECE, 확률 보정 |
| 06-1 | `uv run python -m limitup.collectors.dart` | 이벤트 직전 공시 목록·본문 (`DART_API_KEY` 필요) |
| 06-2 | `uv run python -m limitup.llm.material --sample 50` | LLM 재료 분석 JSON (`OPENAI_*` 필요) |
| 06-4 | `uv run python -m limitup.llm.chart` | AI 분석용 1분봉 차트 |
| 06-5 | `uv run python -m limitup.llm.vision --sample 30` | VLM 차트 분석 JSON |
| 06-3·6 | `uv run python -m limitup.llm.compare` (`--explore`) | M0~M3 비교 |
| 07-1 | `uv run python -m limitup.strategies.scores` | 연도별 워크포워드 점수 (`oos_scores.parquet`) |
| 07장 | `uv run python -m limitup.strategies.design` | 설계 기간(2022~2024) 전략 A·B·C 비용 반영 전 성과 |
| 07-5 | `configs/strategies.toml`, `limitup.strategies.rules` | 전략 규칙 설정과 후보 선택 |
| 08장 | `uv run python -m limitup.backtest.run` | 평가 기간 백테스트: 슬리피지, 체결 모형, 비용, 기간·국면, 스트레스 |

이후 장의 코드는 책 집필에 맞춰 추가됩니다.

## 폴더 구조

```text
ai-limitup-trading-tutorial/
├── src/limitup/
│   ├── collectors/   # 02장: 한국투자증권 API 클라이언트, 종목 마스터, 일봉 수집, 점검
│   ├── events/       # 03장: 상한가 계산, 첫 상한가 탐지
│   ├── features/     # 04장
│   ├── models/       # 05~06장
│   ├── strategies/   # 07장
│   ├── backtest/     # 08장
│   ├── execution/    # 09장
│   ├── paths.py
│   └── storage.py    # 02-4: DuckDB
├── data/             # 수집 데이터 (git 제외, sample 만 포함)
├── notebooks/
├── tests/
└── .env.example
```

## 데이터 이용 안내

원시 시세·공시·뉴스 데이터는 이용 조건 때문에 저장소에 포함하지 않습니다. 각자 발급받은 API 키로 수집 스크립트를 실행해 내려받습니다.

## 라이선스

코드는 MIT 라이선스입니다. 책 본문은 위키독스 책의 저작권 정책을 따릅니다.
