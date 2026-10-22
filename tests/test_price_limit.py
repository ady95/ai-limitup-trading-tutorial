import datetime as dt

from limitup.events.price_limit import lower_limit, tick_size, upper_limit

NEW = dt.date(2026, 10, 22)
OLD = dt.date(2022, 6, 1)


def test_upper_limit_new_ticks():
    # 2026-10-22 실측: 한국투자증권 현재가 API의 기준가·상한가
    cases = {10_140: 13_180, 17_680: 22_950, 30_950: 40_200, 63_500: 82_500,
             179_500: 233_000, 223_500: 290_500, 430_000: 559_000, 1_294_000: 1_682_000}
    for base, expected in cases.items():
        assert upper_limit(base, NEW) == expected


def test_old_tick_tables_differ_by_market():
    assert tick_size(150_000, OLD, "KOSPI") == 500
    assert tick_size(150_000, OLD, "KOSDAQ") == 100
    assert tick_size(1_500, OLD, "KOSPI") == 5
    assert tick_size(1_500, NEW) == 1


def test_lower_limit_rounds_up():
    assert lower_limit(223_500, NEW) == 156_500     # 제한폭 67,000 -> 156,500


def test_limit_width_uses_base_price_tick():
    # 03-1 실측: 기준가 21,900 (50원 단위) -> 제한폭 6,570 을 6,550 으로 버림 -> 하한가 15,350
    assert lower_limit(21_900, NEW) == 15_350
    assert lower_limit(57_600, NEW, "KOSDAQ") == 40_400
    assert upper_limit(17_680, NEW) == 22_950       # 17,680 + 5,300 = 22,980 -> 50원 단위 버림
