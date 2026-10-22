"""05-6 예측 확률 보정과 모델 평가

    uv run python -m limitup.models.calibrate

1. 신뢰도표: 예측 확률 구간별 평균 예측 대 실제 비율 (평가 기간)
2. Brier Score와 ECE(기대 보정 오차)
3. 검증 기간 예측으로 보정기(Isotonic, Platt)를 학습해 평가 기간에 적용
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from limitup.models.baseline import RuleModel
from limitup.models.common import FEATURES, MODEL_DIR, load, parts

BINS = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]


def ece(y, p, bins=BINS) -> float:
    """구간별 |평균 예측 - 실제 비율| 을 건수로 가중평균"""
    b = pd.cut(p, bins, include_lowest=True)
    g = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=True)
    return float((g.size() * (g.p.mean() - g.y.mean()).abs()).sum() / len(y))


def reliability(y, p, bins=BINS) -> pd.DataFrame:
    b = pd.cut(p, bins, include_lowest=True)
    return pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean"))


def main():
    ds = load()
    tr, va, te = parts(ds)
    m1 = lgb.Booster(model_file=str(MODEL_DIR / "m1_y_close.txt"))
    pv, pt = m1.predict(va[FEATURES]), m1.predict(te[FEATURES])
    yv, yt = va.y_close.to_numpy(), te.y_close.to_numpy()
    p0 = RuleModel().fit(tr).predict_proba(te).to_numpy()

    print(f"평가 기간 {len(te):,}건: 실제 비율 {yt.mean():.1%}, M1 평균 예측 {pt.mean():.1%}, M0 평균 예측 {p0.mean():.1%}")
    print("\n[M1 신뢰도표 — 평가 기간]")
    print(reliability(yt, pt).to_string(formatters={"mean_pred": "{:.1%}".format, "actual": "{:.1%}".format}))

    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yv)
    platt = LogisticRegression().fit(np.log(pv / (1 - pv)).reshape(-1, 1), yv)
    cands = {
        "M0 규칙": p0,
        "M1 원래 확률": pt,
        "M1 + Isotonic(검증)": iso.predict(pt),
        "M1 + Platt(검증)": platt.predict_proba(np.log(pt / (1 - pt)).reshape(-1, 1))[:, 1],
        "기준선(항상 평균)": np.full(len(yt), tr.y_close.mean()),
    }
    rows = [{"모델": k, "평균 예측": v.mean(), "Brier": brier_score_loss(yt, v), "ECE": ece(yt, v)}
            for k, v in cands.items()]
    print("\n[평가 기간 확률 품질]")
    print(pd.DataFrame(rows).to_string(index=False, formatters={"평균 예측": "{:.1%}".format,
                                                              "Brier": "{:.4f}".format, "ECE": "{:.4f}".format}))
    # 기간별 기준선 변화: 평가 기간을 반기로 나눠 실제 비율과 평균 예측
    h = te.assign(p=pt, half=te.date.dt.year.astype(str) + np.where(te.date.dt.month <= 6, "H1", "H2"))
    print("\n[반기별 실제 비율 대 평균 예측]")
    print(h.groupby("half").agg(n=("p", "size"), actual=("y_close", "mean"), mean_pred=("p", "mean")).to_string(
        formatters={"actual": "{:.1%}".format, "mean_pred": "{:.1%}".format}))


if __name__ == "__main__":
    main()
