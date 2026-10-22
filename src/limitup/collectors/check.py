"""02-5 일봉 데이터 정합성 점검

    uv run python -m limitup.collectors.check

1. 수집 범위: 모집단 중 데이터가 없는 종목
2. 누락일: 종목의 첫 거래일~마지막 거래일 사이에 시장 거래일인데 행이 없는 날
3. 가격 이상값: 고가 < 저가, 시가·종가가 고가·저가 범위 밖, 0 이하 가격
4. 거래정지일: 거래량 0 인 행 (휴장일은 행 자체가 없다)
5. 기준가격 변경: 오늘 기준가 != 어제 종가 → 권리락·액면분할 등 수정 사건 후보
6. 시장 구조 변화: 기간별로 [5]의 비율과 호가단위에 맞지 않는 가격의 비율을 비교
   - 2025-03-04: 넥스트레이드 출범, KRX 중간가 호가 도입
   - 2026-09-14: KRX 애프터마켓 개설 → 일봉에 16:00~20:00 체결이 섞이는지 확인
"""

from limitup.storage import build

# 기간 경계: 시장 구조가 바뀐 날
PERIODS = """CASE WHEN date < DATE '2025-03-04' THEN '1) ~2025-03-03'
                  WHEN date < DATE '2026-09-14' THEN '2) 2025-03-04~2026-09-11'
                  ELSE '3) 2026-09-14~' END"""
# 호가단위 (2023-01-25 개편 전에는 시장별로 다름). 시장은 현재 소속 기준
TICK = """CASE WHEN date >= DATE '2023-01-25' THEN
                  CASE WHEN p < 2000 THEN 1 WHEN p < 5000 THEN 5 WHEN p < 20000 THEN 10
                       WHEN p < 50000 THEN 50 WHEN p < 200000 THEN 100 WHEN p < 500000 THEN 500 ELSE 1000 END
              WHEN market = 'KOSDAQ' THEN
                  CASE WHEN p < 1000 THEN 1 WHEN p < 5000 THEN 5 WHEN p < 10000 THEN 10
                       WHEN p < 50000 THEN 50 ELSE 100 END
              ELSE CASE WHEN p < 1000 THEN 1 WHEN p < 5000 THEN 5 WHEN p < 10000 THEN 10
                       WHEN p < 50000 THEN 50 WHEN p < 100000 THEN 100 WHEN p < 500000 THEN 500 ELSE 1000 END
         END"""


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

    print("[6] 기간별 비교: 전일 종가 != 기준가 비율, 호가단위 밖 가격 비율 (당시 시장 기준)")
    tick = lambda col: TICK.replace("p <", f"{col} <")
    print(con.sql(f"""
        WITH d AS (
            SELECT *, lag(close) OVER (PARTITION BY code ORDER BY date) AS prev_close
            FROM market_daily WHERE volume > 0)
        SELECT {PERIODS} AS period, count(*) AS rows,
               round(100 * avg((prev_close IS NOT NULL AND base != prev_close)::INT), 2) AS base_gap_pct,
               round(100 * avg((close % ({tick('close')}) != 0)::INT), 2) AS offtick_close_pct,
               round(100 * avg((high % ({tick('high')}) != 0 OR low % ({tick('low')}) != 0)::INT), 2)
                   AS offtick_high_low_pct
        FROM d GROUP BY period ORDER BY period"""))

    print("[7] 시장 이전: 현재 시장 상장일 이전의 행 (이전 시장의 시세)")
    print(con.sql("""
        SELECT u.market AS now, CASE WHEN u.market = 'KOSPI' THEN 'KOSDAQ' ELSE 'KONEX' END AS before,
               count(DISTINCT code) AS stocks, count(*) AS rows
        FROM daily d JOIN universe u USING (code)
        WHERE u.listed IS NOT NULL AND d.date < u.listed GROUP BY ALL ORDER BY now"""))


if __name__ == "__main__":
    main()
