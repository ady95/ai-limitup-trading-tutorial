"""09-5 주문 안전장치 — 비상 중지, 주문 한도, 중복 방지, 기록

    from limitup.execution.guard import Guard
    g = Guard(max_orders=5, max_amount=2_000_000)
    if g.allow(code, amount): ...주문... ; g.record(...)

- 비상 중지: 저장소의 STOP 파일이 있으면 어떤 주문도 내지 않는다 (touch STOP 으로 즉시 중지)
- 한도: 하루 주문 수, 주문 1건 금액, 하루 총액
- 중복 방지: 같은 날 같은 종목에 두 번 주문하지 않는다. 주문을 보내기 전에 예약(pending)을 먼저 기록하므로,
  응답을 받기 전에 프로그램이 죽어도 재시작 때 같은 종목을 다시 주문하지 않는다
- 동시 실행 방지: 같은 날 Guard 는 한 프로세스만 만들 수 있다 (<날짜>_guard.lock)
- 계좌: 모의투자 앱키로만 주문한다 (Guard 는 KIS("paper") 가 아니면 거부)
- 모든 판단을 data/processed/paper_orders/<날짜>_guard.jsonl 에 남긴다
"""

import atexit
import datetime as dt
import json
import os

from limitup.paths import PROCESSED, ROOT

STOP_FILE = ROOT / "STOP"


class Guard:
    def __init__(self, kis, max_orders=5, max_amount=2_000_000, max_total=10_000_000):
        if kis.env != "paper":
            raise RuntimeError("이 책의 주문 실습은 모의투자 계좌에서만 실행합니다")
        self.max_orders, self.max_amount, self.max_total = max_orders, max_amount, max_total
        self.day = dt.date.today().strftime("%Y%m%d")
        self.log_path = PROCESSED / "paper_orders" / f"{self.day}_guard.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.log_path.with_suffix(".lock")
        try:
            os.close(os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            raise RuntimeError(f"{self.lock_path} 가 있습니다 — 다른 실행이 진행 중이거나 이전 실행이 비정상 종료됐습니다. "
                               "증권사 주문 내역을 확인한 뒤 이 파일을 지우세요") from None
        atexit.register(self.close)
        self.done = self._load()

    def close(self):
        """잠금 해제 (프로세스가 정상 종료하면 자동으로 불린다)"""
        self.lock_path.unlink(missing_ok=True)

    def _load(self):
        if not self.log_path.exists():
            return []
        recs = [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines()]
        done = {r["code"]: r for r in recs if r.get("sent") in (True, "pending")}   # 예약만 남은 주문도 낸 것으로 본다
        return list(done.values())

    def _write(self, rec):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def allow(self, code: str, amount: int) -> tuple[bool, str]:
        if STOP_FILE.exists():
            reason = "비상 중지 파일(STOP)이 있음"
        elif len(self.done) >= self.max_orders:
            reason = f"하루 주문 수 한도 {self.max_orders}건 도달"
        elif amount > self.max_amount:
            reason = f"주문 금액 {amount:,}원이 한도 {self.max_amount:,}원 초과"
        elif sum(d["amount"] for d in self.done) + amount > self.max_total:
            reason = f"하루 총액 한도 {self.max_total:,}원 초과"
        elif any(d["code"] == code for d in self.done):
            reason = "오늘 이미 주문한 종목"
        else:
            return True, "허용"
        self._write({"time": dt.datetime.now().isoformat(timespec="seconds"), "code": code,
                     "amount": amount, "sent": False, "reason": reason})
        return False, reason

    def reserve(self, code: str, amount: int):
        """주문을 보내기 직전에 호출한다. 이후 응답이 유실돼도 같은 종목은 다시 주문하지 않는다."""
        rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "code": code, "amount": amount,
               "sent": "pending"}
        self._write(rec)
        self.done.append(rec)

    def record(self, code: str, amount: int, response: dict):
        rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "code": code, "amount": amount,
               "sent": True, "rt_cd": response.get("rt_cd"), "msg": response.get("msg1", "").strip(),
               "order_no": (response.get("output") or {}).get("ODNO")}
        self._write(rec)
        self.done = [d for d in self.done if d["code"] != code] + [rec]
