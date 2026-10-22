"""06-3, 06-6 M0~M3 비교

    uv run python -m limitup.llm.compare --explore   # 학습 기간: 재료 유형별 연속 상한가 비율
    uv run python -m limitup.llm.compare             # 전체 기간(M0·M1·M2), 최근 1년(M1~M3) 비교

- M0: 규칙 4개 (05-1)
- M1: LightGBM + 일봉·시장 피처 21개 (05-2)
- M2: M1 + LLM 재료 피처
- M3: M2 + VLM 차트 피처 (차트는 분봉이 있는 최근 1년만)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from limitup.llm.features import LLM_FEATURES, VLM_FEATURES, llm_features, vlm_features
from limitup.models.baseline import RuleModel
from limitup.models.common import FEATURES, cluster_boot, fmt, load, metrics, parts
from limitup.features.intraday import INTRADAY_FEATURES
from limitup.models.lgbm import PARAMS, fit


def dataset():
    return vlm_features(llm_features(load()))


def explore(ds):
    tr = ds[ds.split == "train"]
    print(f"학습 기간 {len(tr):,}건")
    t = tr.groupby("l_group").agg(n=("y_close", "size"), y_close=("y_close", "mean"),
                                  y_touch=("y_touch", "mean"), ret_oc=("y_ret_oc", "mean"))
    print(t.sort_values("n", ascending=False).to_string(
        formatters={"y_close": "{:.1%}".format, "y_touch": "{:.1%}".format, "ret_oc": "{:.1%}".format}))
    for col in ("l_impact", "l_duration", "l_uncertainty"):
        g = tr.groupby(col).agg(n=("y_close", "size"), y_close=("y_close", "mean"))
        print(f"\n[{col}]\n" + g.to_string(formatters={"y_close": "{:.1%}".format}))


def full_period(ds):
    tr, va, te = parts(ds)
    m0 = RuleModel().fit(tr)
    m1 = fit(tr, va, FEATURES)
    m2 = fit(tr, va, FEATURES + LLM_FEATURES)
    d = te.assign(m0=m0.predict_proba(te),
                  m1=m1.predict(te[FEATURES], num_iteration=m1.best_iteration),
                  m2=m2.predict(te[FEATURES + LLM_FEATURES], num_iteration=m2.best_iteration))
    print(f"[전체 기간 평가 {len(te):,}건]  반복 수 M1 {m1.best_iteration}, M2 {m2.best_iteration}")
    for k in ("m0", "m1", "m2"):
        print(f"  {k.upper()}: " + fmt(metrics(d, k)))
    diff = lambda b: average_precision_score(b.y_close, b.m2) - average_precision_score(b.y_close, b.m1)
    lo, hi = cluster_boot(d, diff)
    print(f"  PR-AUC 차이 (M2 - M1) {diff(d):+.3f}, 날짜 단위 95% 구간 {lo:+.3f} ~ {hi:+.3f}")
    imp = pd.Series(m2.feature_importance("gain"), index=FEATURES + LLM_FEATURES)
    print("  M2 gain 상위:", (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict())


def recent_year(ds, seeds=5):
    tr = ds[(ds.i_split == "train") & ~ds.i_purge]
    te = ds[ds.i_split == "test"]
    small = {**PARAMS, "num_leaves": 7, "min_child_samples": 30}
    sets = {"M1 일봉·시장": FEATURES, "M1 + 장중 수치": FEATURES + INTRADAY_FEATURES,
            "M2 (+LLM)": FEATURES + LLM_FEATURES, "M3 (+LLM+VLM)": FEATURES + LLM_FEATURES + VLM_FEATURES,
            "M3 + 장중 수치": FEATURES + LLM_FEATURES + VLM_FEATURES + INTRADAY_FEATURES}
    d = te.copy()
    for name, feats in sets.items():
        p = np.zeros(len(te))
        for s in range(seeds):
            p += fit(tr, features=feats, rounds=300, params={**small, "seed": s}).predict(te[feats]) / seeds
        d[name] = p
    d["M0 규칙"] = RuleModel().fit(tr).predict_proba(te)
    print(f"\n[최근 1년: 장중 학습 {len(tr)}건 → 장중 평가 {len(te)}건]")
    for name in ["M0 규칙", *sets]:
        print(f"  {name:14s}: " + fmt(metrics(d, name)))
    for a, b in [("M2 (+LLM)", "M1 일봉·시장"), ("M3 (+LLM+VLM)", "M2 (+LLM)"), ("M1 + 장중 수치", "M1 일봉·시장")]:
        diff = lambda x, a=a, b=b: average_precision_score(x.y_close, x[a]) - average_precision_score(x.y_close, x[b])
        lo, hi = cluster_boot(d, diff)
        print(f"  PR-AUC 차이 ({a} - {b}) {diff(d):+.3f}, 95% 구간 {lo:+.3f} ~ {hi:+.3f}")
    vt = ds[ds.v_pattern.notna()]
    print("\n[VLM 패턴 대 수치 유형 (분봉 있는 전체)]")
    print(pd.crosstab(vt.i_type, vt.v_pattern).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    a = ap.parse_args()
    ds = dataset()
    if a.explore:
        explore(ds)
    else:
        full_period(ds)
        recent_year(ds)
