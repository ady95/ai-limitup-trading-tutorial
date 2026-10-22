"""05-2, 05-3 LightGBM 분류 모델 M1 — 다음 날 상한가 마감(P2)·도달(P1) 확률

    uv run python -m limitup.models.lgbm               # 학습 → 검증 조기 종료 → 평가
    uv run python -m limitup.models.lgbm --walk        # 연도별 워크포워드
    uv run python -m limitup.models.lgbm --intraday    # 장중 피처 추가 효과 (최근 1년)
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from limitup.models.baseline import RuleModel
from limitup.models.common import (ALL_FEATURES, FEATURES, MODEL_DIR, cluster_boot, fmt, load,
                                   metrics, parts)

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=42)


def fit(tr, va=None, features=FEATURES, y="y_close", rounds=2000, params=PARAMS):
    """검증 구간이 있으면 PR-AUC(average_precision) 기준으로 조기 종료한다."""
    dtr = lgb.Dataset(tr[features], tr[y])
    if va is None:
        return lgb.train(params, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(va[features], va[y], reference=dtr)
    return lgb.train({**params, "metric": "average_precision"}, dtr, num_boost_round=rounds,
                     valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def main_fixed():
    ds = load()
    tr, va, te = parts(ds)
    m0 = RuleModel().fit(tr)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for y in ("y_close", "y_touch"):
        m1 = fit(tr, va, y=y)
        m1.save_model(str(MODEL_DIR / f"m1_{y}.txt"))
        print(f"\n[{y}] 조기 종료 반복 수 {m1.best_iteration}")
        for name, d in [("검증", va), ("평가", te)]:
            d = d.assign(m0=m0.predict_proba(d), m1=m1.predict(d[FEATURES], num_iteration=m1.best_iteration))
            print(f"  M0 {name}: " + fmt(metrics(d, "m0", y)))
            print(f"  M1 {name}: " + fmt(metrics(d, "m1", y)))
        if y == "y_close":
            d = te.assign(m0=m0.predict_proba(te), m1=m1.predict(te[FEATURES], num_iteration=m1.best_iteration))
            diff = lambda b: average_precision_score(b.y_close, b.m1) - average_precision_score(b.y_close, b.m0)
            lo, hi = cluster_boot(d, diff)
            print(f"  평가 PR-AUC 차이 (M1 - M0) {diff(d):+.3f}, 날짜 단위 95% 구간 {lo:+.3f} ~ {hi:+.3f}")
            d[["code", "date", "y_close", "m0", "m1"]].to_parquet(MODEL_DIR / "pred_test.parquet", index=False)


def main_walk():
    """학습 기간을 1년씩 늘리며 다음 기간을 평가한다. 반복 수는 고정 분할의 조기 종료 값을 쓴다."""
    ds = load()
    tr, va, _ = parts(ds)
    rounds = fit(tr, va).best_iteration
    folds = [("2023", "2023-01-01", "2024-01-01"), ("2024", "2024-01-01", "2025-01-01"),
             ("2025~", "2025-01-01", "2100-01-01")]
    rows = []
    for name, start, end in folds:
        train = ds[(ds.date < start) & (ds.t3_date.fillna(ds.date) < start)]   # 경계 퍼지
        test = ds[(ds.date >= start) & (ds.date < end)]
        m0 = RuleModel().fit(train)
        m1 = fit(train, rounds=rounds)
        d = test.assign(m0=m0.predict_proba(test), m1=m1.predict(test[FEATURES]))
        a, b = metrics(d, "m0"), metrics(d, "m1")
        rows.append({"평가": name, "학습": f"~{int(start[:4]) - 1}", "n": len(test), "기준선": a["base"],
                     "M0 PR-AUC": a["pr_auc"], "M1 PR-AUC": b["pr_auc"],
                     "M0 상위10%": a["p_at_10"], "M1 상위10%": b["p_at_10"]})
    t = pd.DataFrame(rows)
    print(f"반복 수 {rounds}")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main_intraday(seeds=5):
    """최근 1년 안에서 같은 학습·평가 구간으로 '일봉만' 대 '일봉+장중' 비교"""
    ds = load()
    tr = ds[(ds.i_split == "train") & ~ds.i_purge]
    te = ds[ds.i_split == "test"]
    small = {**PARAMS, "num_leaves": 7, "min_child_samples": 30}
    pa, pb = np.zeros(len(te)), np.zeros(len(te))
    for s in range(seeds):                                  # 작은 표본이라 시드 평균
        p = {**small, "seed": s}
        pa += fit(tr, features=FEATURES, rounds=300, params=p).predict(te[FEATURES]) / seeds
        pb += fit(tr, features=ALL_FEATURES, rounds=300, params=p).predict(te[ALL_FEATURES]) / seeds
    d = te.assign(daily=pa, both=pb)
    print(f"장중 학습 {len(tr)}건 → 장중 평가 {len(te)}건")
    print("  일봉+시장 피처   : " + fmt(metrics(d, "daily")))
    print("  + 장중 피처     : " + fmt(metrics(d, "both")))
    diff = lambda b: average_precision_score(b.y_close, b.both) - average_precision_score(b.y_close, b.daily)
    lo, hi = cluster_boot(d, diff)
    print(f"  PR-AUC 차이 (장중 추가 - 일봉만) {diff(d):+.3f}, 날짜 단위 95% 구간 {lo:+.3f} ~ {hi:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--intraday", action="store_true")
    a = ap.parse_args()
    if a.walk:
        main_walk()
    elif a.intraday:
        main_intraday()
    else:
        main_fixed()
