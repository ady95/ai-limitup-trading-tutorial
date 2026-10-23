import types

import pytest

import limitup.execution.guard as g


def make(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(g, "PROCESSED", tmp_path)
    monkeypatch.setattr(g, "STOP_FILE", tmp_path / "STOP")
    return g.Guard(types.SimpleNamespace(env="paper"), **kw)


def test_real_account_refused():
    with pytest.raises(RuntimeError):
        g.Guard(types.SimpleNamespace(env="real"))


def test_limits_duplicates_and_stop(tmp_path, monkeypatch):
    guard = make(tmp_path, monkeypatch, max_orders=2, max_amount=2_000_000, max_total=3_000_000)
    assert guard.allow("000001", 1_500_000)[0]
    guard.record("000001", 1_500_000, {"rt_cd": "0"})
    assert not guard.allow("000001", 1_000_000)[0]           # 중복
    assert not guard.allow("000002", 2_500_000)[0]           # 1건 한도
    assert not guard.allow("000002", 1_600_000)[0]           # 총액 한도
    (tmp_path / "STOP").write_text("")
    assert not guard.allow("000003", 100_000)[0]             # 비상 중지
