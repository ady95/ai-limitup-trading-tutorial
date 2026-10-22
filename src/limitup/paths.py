"""저장소 안의 경로를 한곳에서 정한다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
DAILY_DIR = RAW / "daily"              # 종목별 일봉 Parquet (02-3)
UNIVERSE = RAW / "universe.parquet"    # 상장·폐지 종목 모집단 (02-3)
INDUSTRY = RAW / "industry.parquet"    # 업종·주요 제품 (04-3)
PROCESSED = DATA / "processed"
DB_PATH = DATA / "limitup.duckdb"      # 02-4
CACHE = ROOT / ".cache"                # 접근토큰 캐시
