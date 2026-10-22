"""07-1 전략에 쓸 모델 점수 — 연도마다 그 이전 데이터로만 학습한 예측(워크포워드)

    uv run python -m limitup.strategies.scores

각 연도(2022, 2023, 2024, 2025~)의 이벤트에 대해 그 연도 이전 이벤트로만 학습한 모델의 예측을 붙인다.
전략 규칙은 2021~2024 (설계 기간)에서 정하고, 2025~ (평가 기간)는 8장에서 한 번만 쓴다.
"""

import lightgbm as lgb
import pandas as pd

from limitup.models.common import FEATURES, MODEL_DIR, load
from limitup.models.lgbm import PARAMS
from limitup.models.regress import BASE

SCORES = MODEL_DIR / "oos_scores.parquet"
YEARS = [("2022", "2022-01-01", "2023-01-01"), ("2023", "2023-01-01", "2024-01-01"),
         ("2024", "2024-01-01", "2025-01-01"), ("2025~", "2025-01-01", "2100-01-01")]
ROUNDS = {"p2": 67, "p1": 47, "gap": 200, "oc": 200, "low_q10": 200}   # 05장 조기 종료 값 근처로 고정


def train(df, y, params, rounds):
    return lgb.train(params, lgb.Dataset(df[FEATURES], df[y]), num_boost_round=rounds)


def main():
    ds = load()
    parts = []
    for name, start, end in YEARS:
        tr = ds[(ds.date < start) & (ds.t3_date.fillna(ds.date) < start)]       # 경계 퍼지
        te = ds[(ds.date >= start) & (ds.date < end)].copy()
        te["p2"] = train(tr, "y_close", PARAMS, ROUNDS["p2"]).predict(te[FEATURES])
        te["p1"] = train(tr, "y_touch", PARAMS, ROUNDS["p1"]).predict(te[FEATURES])
        te["pred_gap"] = train(tr, "y_ret_open", {**BASE, "objective": "huber", "alpha": 0.05},
                               ROUNDS["gap"]).predict(te[FEATURES])
        te["pred_oc"] = train(tr, "y_ret_oc", {**BASE, "objective": "huber", "alpha": 0.05},
                              ROUNDS["oc"]).predict(te[FEATURES])
        te["pred_low_q10"] = train(tr, "y_ret_low", {**BASE, "objective": "quantile", "alpha": 0.1},
                                   ROUNDS["low_q10"]).predict(te[FEATURES])
        te["score_year"] = name
        parts.append(te[["code", "date", "score_year", "p2", "p1", "pred_gap", "pred_oc", "pred_low_q10"]])
        print(f"{name}: 학습 {len(tr):,}건 → 예측 {len(te):,}건")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(SCORES, index=False)
    print(f"→ {SCORES}")


if __name__ == "__main__":
    main()
