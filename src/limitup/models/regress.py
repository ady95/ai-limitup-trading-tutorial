"""05-4 기대수익률과 하방 위험 예측 — LightGBM 회귀·분위수 회귀

    uv run python -m limitup.models.regress

- y_ret_open : T+1 시가 갭 (T일 종가 → T+1 시가)
- y_ret_oc   : T+1 시가 매수 → 종가 매도
- y_ret_low  : T+1 저가 (하방 위험) — 하위 10% 분위수를 예측
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from limitup.models.common import FEATURES, MODEL_DIR, load, parts

BASE = dict(learning_rate=0.03, num_leaves=15, min_child_samples=40, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
TARGETS = {
    "y_ret_open": dict(objective="huber", alpha=0.05),         # 이상값(연속 상한가)에 덜 끌려가게
    "y_ret_oc": dict(objective="huber", alpha=0.05),
    "y_ret_low_q10": dict(objective="quantile", alpha=0.1),    # 하위 10% 분위수
}


def fit_reg(tr, va, y, extra):
    label = "y_ret_low" if y.endswith("_q10") else y
    dtr = lgb.Dataset(tr[FEATURES], tr[label])
    dva = lgb.Dataset(va[FEATURES], va[label], reference=dtr)
    return lgb.train({**BASE, **extra}, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def deciles(d, pred, actual):
    """예측값 10분위별 실제 평균"""
    q = pd.qcut(d[pred].rank(method="first"), 10, labels=range(1, 11))
    return d.groupby(q, observed=True)[actual].mean()


def main():
    ds = load()
    tr, va, te = parts(ds)
    te = te.copy()
    for y, extra in TARGETS.items():
        m = fit_reg(tr, va, y, extra)
        m.save_model(str(MODEL_DIR / f"reg_{y}.txt"))
        te[f"pred_{y}"] = m.predict(te[FEATURES], num_iteration=m.best_iteration)
    print(f"평가 기간 {len(te):,}건")
    for y in ("y_ret_open", "y_ret_oc"):
        rho = spearmanr(te[f"pred_{y}"], te[y]).statistic
        dec = deciles(te, f"pred_{y}", y)
        print(f"\n[{y}] 순위 상관(스피어만) {rho:.3f}")
        print("  예측 10분위별 실제 평균: " + "  ".join(f"{i}:{v:+.1%}" for i, v in dec.items()))
    cover = (te.y_ret_low < te.pred_y_ret_low_q10).mean()
    print(f"\n[y_ret_low 하위 10% 분위수] 실제 저가가 예측 분위수보다 낮았던 비율 {cover:.1%} (목표 10%)")
    dec = deciles(te, "pred_y_ret_low_q10", "y_ret_low")
    print("  예측 분위수 10분위별 실제 저가 평균: " + "  ".join(f"{i}:{v:+.1%}" for i, v in dec.items()))

    # 상한가 확률(05-3)과 함께 보기
    p = pd.read_parquet(MODEL_DIR / "pred_test.parquet")[["code", "date", "m1"]]
    te = te.merge(p, on=["code", "date"])
    top = te[te.m1 >= te.m1.quantile(0.9)]
    print(f"\nP2 확률 상위 10% ({len(top)}건): 연속 상한가 {top.y_close.mean():.1%}, "
          f"T+1 시가 {top.y_ret_open.mean():+.1%}, 시가→종가 {top.y_ret_oc.mean():+.1%}, 저가 {top.y_ret_low.mean():+.1%}")
    rest = te[te.m1 < te.m1.quantile(0.9)]
    print(f"나머지 90% ({len(rest)}건): 연속 상한가 {rest.y_close.mean():.1%}, "
          f"T+1 시가 {rest.y_ret_open.mean():+.1%}, 시가→종가 {rest.y_ret_oc.mean():+.1%}, 저가 {rest.y_ret_low.mean():+.1%}")
    both = te[(te.m1 >= te.m1.quantile(0.9)) & (te.pred_y_ret_oc >= te.pred_y_ret_oc.quantile(0.5))]
    print(f"P2 상위 10% 이면서 시가→종가 예측 상위 50% ({len(both)}건): 시가→종가 {both.y_ret_oc.mean():+.1%}, "
          f"수익 비율 {(both.y_ret_oc > 0).mean():.1%}")
    te[["code", "date", *[c for c in te.columns if c.startswith("pred_")]]].to_parquet(
        MODEL_DIR / "pred_reg_test.parquet", index=False)


if __name__ == "__main__":
    main()
