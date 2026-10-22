"""02-4 DuckDB로 종목별 Parquet 파일을 하나의 테이블처럼 질의하기

    uv run python -m limitup.storage          # 뷰를 만들고 요약을 출력

    from limitup.storage import connect
    con = connect()
    con.sql("SELECT count(*) FROM daily").show()
"""

import duckdb

from limitup.paths import DAILY_DIR, DB_PATH, UNIVERSE


def connect(read_only=False):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def build():
    """Parquet 파일 위에 뷰를 만든다. 데이터를 복사하지 않으므로 수집 후 다시 만들 필요가 없다."""
    con = connect()
    con.execute(f"""CREATE OR REPLACE VIEW daily AS
        SELECT * FROM read_parquet('{DAILY_DIR.as_posix()}/*.parquet')""")
    con.execute(f"""CREATE OR REPLACE VIEW universe AS
        SELECT * FROM read_parquet('{UNIVERSE.as_posix()}')""")
    # 02-5 당시 시장(market) 열을 붙이고 코넥스 시절 행을 뺀 분석용 뷰.
    # 마스터의 상장일(listed)은 '현재 시장'에 상장한 날이므로, 그 전 행은 이전 시장의 시세다.
    #   현재 KOSPI → 이전은 KOSDAQ (이전상장),  현재 KOSDAQ → 이전은 KONEX (가격제한폭 ±15%)
    con.execute("""CREATE OR REPLACE VIEW market_daily AS
        SELECT * EXCLUDE (market_at), market_at AS market FROM (
            SELECT d.*, CASE WHEN u.listed IS NOT NULL AND d.date < u.listed
                             THEN CASE WHEN u.market = 'KOSPI' THEN 'KOSDAQ' ELSE 'KONEX' END
                             ELSE u.market END AS market_at
            FROM daily d JOIN universe u USING (code))
        WHERE market_at IN ('KOSPI', 'KOSDAQ')""")
    return con


def main():
    con = build()
    print(con.sql("""
        SELECT count(*) AS rows, count(DISTINCT code) AS stocks,
               min(date)::DATE AS first, max(date)::DATE AS last,
               count(DISTINCT date) AS trading_days
        FROM daily"""))
    print(con.sql("""
        SELECT u.market, u.delisted IS NOT NULL AS delisted,
               count(DISTINCT d.code) AS stocks, count(*) AS rows
        FROM daily d JOIN universe u USING (code)
        GROUP BY ALL ORDER BY ALL"""))
    size = sum(p.stat().st_size for p in DAILY_DIR.glob("*.parquet"))
    print(f"Parquet {len(list(DAILY_DIR.glob('*.parquet'))):,}개, {size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
