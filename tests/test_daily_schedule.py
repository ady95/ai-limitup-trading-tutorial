import datetime as dt
import types

import limitup.execution.daily as d


class FakeKIS:
    """휴장일 조회 응답 흉내: 기준일부터 여러 날의 개장 여부"""
    def __init__(self, days):
        self.days = days

    def get(self, path, tr_id, params):
        return {"output": [{"bass_dt": k, "opnd_yn": v} for k, v in self.days.items()]}


def test_next_open_day_skips_holidays():
    kis = FakeKIS({"20261118": "Y", "20261119": "Y", "20261120": "Y"})
    assert d.next_open_day(kis, dt.date(2026, 11, 17)) == dt.date(2026, 11, 18)
    kis = FakeKIS({"20261003": "N", "20261004": "N", "20261005": "Y"})
    assert d.next_open_day(kis, dt.date(2026, 10, 2)) == dt.date(2026, 10, 5)


def test_on_time_rejects_late_start(monkeypatch):
    monkeypatch.setattr(d, "wait_until", lambda hhmmss: None)
    class Late(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 26, 15, 16, 0)
    monkeypatch.setattr(d, "dt", types.SimpleNamespace(datetime=Late, time=dt.time, date=dt.date, timedelta=dt.timedelta))
    assert not d.on_time("15:10:00", "15:14:00")
    assert d.on_time("15:10:00", "15:20:00")
