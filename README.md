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
