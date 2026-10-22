"""08-1 이벤트 기반 백테스트 엔진

사건(주문 제출, 체결, 청산)을 시각 순서대로 처리하며 현금·포지션·손익을 기록한다.
전략은 '주문 계획'(언제 무엇을 얼마에 사고팔지)만 만들고, 체결 여부는 체결 모형이 정한다.

    from limitup.backtest.engine import Engine, Order
    eng = Engine(capital=100_000_000)
    eng.submit(Order(...)); eng.run()
    eng.trades, eng.equity
"""

import heapq
import itertools
from dataclasses import dataclass, field

import pandas as pd

from limitup.backtest.costs import COMMISSION, tax_rate


@dataclass(order=True)
class Event:
    time: pd.Timestamp
    seq: int
    kind: str = field(compare=False)           # "entry" | "exit"
    order: "Order" = field(compare=False)


@dataclass
class Order:
    code: str
    entry_time: pd.Timestamp
    entry_price: float          # 체결 모형이 정한 매수 체결가 (슬리피지 포함)
    exit_time: pd.Timestamp
    exit_price: float           # 매도 체결가 (슬리피지 포함)
    amount: float               # 매수 금액(원)
    filled: bool = True         # 체결 모형의 판정. False면 주문만 내고 체결되지 않음
    tag: str = ""


class Engine:
    def __init__(self, capital: float, commission: float = COMMISSION):
        self.cash = capital
        self.capital = capital
        self.commission = commission
        self.queue, self.seq = [], itertools.count()
        self.positions = {}
        self.trades, self.equity, self.unfilled = [], [], []

    def submit(self, o: Order):
        heapq.heappush(self.queue, Event(o.entry_time, next(self.seq), "entry", o))

    def run(self):
        while self.queue:
            ev = heapq.heappop(self.queue)
            o = ev.order
            if ev.kind == "entry":
                if not o.filled:
                    self.unfilled.append(o)
                    continue
                qty = int(o.amount // o.entry_price)
                cost = qty * o.entry_price * (1 + self.commission)
                if qty == 0 or cost > self.cash:
                    self.unfilled.append(o)            # 현금 부족도 미체결로 처리
                    continue
                self.cash -= cost
                self.positions[(o.code, o.entry_time)] = (qty, cost)
                heapq.heappush(self.queue, Event(o.exit_time, next(self.seq), "exit", o))
            else:
                qty, cost = self.positions.pop((o.code, o.entry_time))
                proceeds = qty * o.exit_price * (1 - self.commission - tax_rate(o.exit_time))
                self.cash += proceeds
                self.trades.append({"code": o.code, "entry": o.entry_time, "exit": o.exit_time, "qty": qty,
                                    "cost": cost, "proceeds": proceeds, "pnl": proceeds - cost,
                                    "ret": proceeds / cost - 1, "tag": o.tag})
                held = sum(c for _, c in self.positions.values())      # 보유분은 매수 원가로 평가
                self.equity.append({"time": o.exit_time, "equity": self.cash + held})
        return self

    def summary(self) -> dict:
        t = pd.DataFrame(self.trades)
        n_orders = len(t) + len(self.unfilled)
        if t.empty:
            return {"orders": n_orders, "trades": 0, "fill_rate": 0.0}
        eq = pd.concat([pd.Series([self.capital]), pd.DataFrame(self.equity).equity], ignore_index=True)
        peak = eq.cummax()                                      # 초기 자본부터 최고점 추적
        win, loss = t.ret[t.ret > 0], t.ret[t.ret <= 0]
        return {"orders": n_orders, "trades": len(t), "fill_rate": len(t) / n_orders,
                "mean_ret": t.ret.mean(), "win_rate": (t.ret > 0).mean(),
                "payoff": win.mean() / -loss.mean() if len(loss) and len(win) else float("nan"),
                "total_pnl": t.pnl.sum(), "total_return": eq.iloc[-1] / self.capital - 1,
                "mdd": (eq / peak - 1).min()}
