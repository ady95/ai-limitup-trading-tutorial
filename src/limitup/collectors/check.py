"""02-5 일봉 데이터 정합성 점검

    uv run python -m limitup.collectors.check

1. 수집 범위: 모집단 중 데이터가 없는 종목
2. 누락일: 종목의 첫 거래일~마지막 거래일 사이에 시장 거래일인데 행이 없는 날
3. 가격 이상값: 고가 < 저가, 시가·종가가 고가·저가 범위 밖, 0 이하 가격
4. 거래정지일: 거래량 0 인 행 (휴장일은 행 자체가 없다)
5. 기준가격 변경: 오늘 기준가 != 어제 종가 → 권리락·액면분할 등 수정 사건 후보
"""

from limitup.storage import build


def main():
    con = build()

    print("[1] 수집 범위")
    print(con.sql("""
        SELECT count(*) AS universe,
               count(*) FILTER (WHERE code NOT IN (SELECT DISTINCT code FROM daily)) AS no_data
        FROM universe"""))

    print("[2] 누락일 (시장 거래일 기준)")
    con.execute("""CREATE OR REPLACE TEMP TABLE cal AS
        SELECT date FROM daily GROUP BY date HAVING count(*) >= 500""")   # 휴장일 오인 방지
    print(con.sql("""
        WITH span AS (SELECT code, min(date) AS d0, max(date) AS d1, count(*) AS n FROM daily GROUP BY code),
             expect AS (SELECT s.code, s.n, count(c.date) AS expected
                        FROM span s JOIN cal c ON c.date BETWEEN s.d0 AND s.d1 GROUP BY s.code, s.n)
        SELECT count(*) FILTER (WHERE expected > n) AS stocks_with_gaps,
               sum(expected - n) FILTER (WHERE expected > n) AS missing_rows,
               count(*) FILTER (WHERE expected < n) AS stocks_with_extra_days
        FROM expect"""))

    print("[3] 가격 이상값")
    print(con.sql("""
        SELECT count(*) FILTER (WHERE high < low) AS high_lt_low,
               count(*) FILTER (WHERE close > high OR close < low) AS close_out,
               count(*) FILTER (WHERE open > high OR open < low) AS open_out,
               count(*) FILTER (WHERE least(open, high, low, close) <= 0) AS nonpositive
        FROM daily"""))

    print("[4] 거래정지일 (거래량 0)")
    print(con.sql("""
        SELECT count(*) AS halted_rows, count(DISTINCT code) AS stocks,
               count(*) FILTER (WHERE open = close AND high = low AND change = 0) AS flat_rows
        FROM daily WHERE volume = 0"""))

    print("[5] 기준가격 != 전일 종가 (수정 사건 후보)")
    con.execute("""CREATE OR REPLACE TEMP TABLE base_gap AS
        SELECT code, date, base, lag(close) OVER (PARTITION BY code ORDER BY date) AS prev_close,
               flng_cls_code, mod_yn
        FROM daily QUALIFY prev_close IS NOT NULL AND base != prev_close""")
    print(con.sql("""SELECT count(*) AS events, count(DISTINCT code) AS stocks,
                     round(median(base / prev_close), 3) AS median_ratio FROM base_gap"""))
    print(con.sql("""SELECT flng_cls_code, mod_yn, count(*) AS n FROM base_gap
                     GROUP BY ALL ORDER BY n DESC LIMIT 8"""))


if __name__ == "__main__":
    main()
