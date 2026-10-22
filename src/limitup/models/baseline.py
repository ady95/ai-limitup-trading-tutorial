"""05-1 규칙 기반 기준 모델 M0

    uv run python -m limitup.models.baseline

04장의 학습 기간 탐색에서 관계가 뚜렷했던 네 가지를 규칙으로 만들고, 만족한 규칙 수(0~4)를 점수로 쓴다.
기준값(분위수)은 학습 기간에서만 정하고, 점수별 확률은 학습 기간의 실제 비율로 정한다.
"""

import pandas as pd

from limitup.models.common import fmt, load, metrics, parts


class RuleModel:
    def fit(self, tr: pd.DataFrame):
        self.th = {
            "small_value": tr.f_value_log.quantile(0.2),     # 거래대금 하위 20%
            "big_gap": tr.f_gap.quantile(0.8),               # 시가 갭 상위 20%
            "recent_jump": tr.f_maxret20.quantile(0.8),      # 직전 20일 최대 상승 상위 20%
            "hot_market": 8,                                 # 당일 상한가 8개 이상
        }
        s = self.score(tr)
        # 점수별 실제 비율 (표본이 적은 점수는 전체 평균 쪽으로 당긴다)
        k, prior = 20, tr.y_close.mean()
        g = tr.groupby(s).y_close.agg(["sum", "count"])
        self.prob = ((g["sum"] + k * prior) / (g["count"] + k)).to_dict()
        self.prior = prior
        return self

    def rules(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "small_value": df.f_value_log <= self.th["small_value"],
            "big_gap": df.f_gap >= self.th["big_gap"],
            "recent_jump": df.f_maxret20 >= self.th["recent_jump"],
            "hot_market": df.m_n_lu >= self.th["hot_market"],
        })

    def score(self, df: pd.DataFrame) -> pd.Series:
        return self.rules(df).sum(axis=1)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return self.score(df).map(self.prob).fillna(self.prior)


def main():
    ds = load()
    tr, va, te = parts(ds)
    m0 = RuleModel().fit(tr)
    print("규칙 기준값 (학습 기간에서 결정):", {k: round(v, 4) for k, v in m0.th.items()})
    for name, d in [("학습", tr), ("검증", va), ("평가", te)]:
        t = d.assign(score=m0.score(d)).groupby("score").agg(n=("y_close", "size"), y_close=("y_close", "mean"))
        print(f"\n[{name}] 규칙 점수별 T+1 상한가 마감\n" + t.to_string(formatters={"y_close": "{:.1%}".format}))
    print()
    for name, d in [("검증", va), ("평가", te)]:
        print(f"M0 {name}: " + fmt(metrics(d.assign(p=m0.predict_proba(d)), "p")))


if __name__ == "__main__":
    main()
