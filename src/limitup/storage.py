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
