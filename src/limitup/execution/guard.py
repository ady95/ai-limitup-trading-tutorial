"""09-5 주문 안전장치 — 비상 중지, 주문 한도, 중복 방지, 기록

    from limitup.execution.guard import Guard
    g = Guard(max_orders=5, max_amount=2_000_000)
    if g.allow(code, amount): ...주문... ; g.record(...)

- 비상 중지: 저장소의 STOP 파일이 있으면 어떤 주문도 내지 않는다 (touch STOP 으로 즉시 중지)
- 한도: 하루 주문 수, 주문 1건 금액, 하루 총액
- 중복 방지: 같은 날 같은 종목에 두 번 주문하지 않는다
- 계좌: 모의투자 앱키로만 주문한다 (Guard 는 KIS("paper") 가 아니면 거부)
- 모든 판단을 data/processed/paper_orders/<날짜>_guard.jsonl 에 남긴다
"""

import datetime as dt
import json

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
        self.done = self._load()

    def _load(self):
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text(encoding="utf-8").splitlines()
                if '"sent": true' in l]

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

    def record(self, code: str, amount: int, response: dict):
        rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "code": code, "amount": amount,
               "sent": True, "rt_cd": response.get("rt_cd"), "msg": response.get("msg1", "").strip(),
               "order_no": (response.get("output") or {}).get("ODNO")}
        self._write(rec)
        self.done.append(rec)
