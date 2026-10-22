"""05-5 SHAP으로 M1(y_close) 해석

    uv run python -m limitup.models.explain

- 전체 중요도: 평가 기간 예측에서 피처별 |SHAP| 평균
- 방향: 피처 값이 클수록 SHAP이 커지는가 (스피어만 상관)
- 개별 설명: 확률이 가장 높았던 이벤트 하나를 피처별 기여로 분해
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from limitup.models.common import FEATURES, MODEL_DIR, load, parts


def main():
    ds = load()
    _, _, te = parts(ds)
    m1 = lgb.Booster(model_file=str(MODEL_DIR / "m1_y_close.txt"))
    explainer = shap.TreeExplainer(m1)
    sv = explainer.shap_values(te[FEATURES])               # 로그 오즈(log-odds) 단위의 기여
    imp = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": np.abs(sv).mean(0),
        "direction": [spearmanr(te[f], sv[:, i], nan_policy="omit").statistic for i, f in enumerate(FEATURES)],
        "gain_share": m1.feature_importance("gain") / m1.feature_importance("gain").sum(),
    }).sort_values("mean_abs_shap", ascending=False)
    print(f"평가 기간 {len(te):,}건, 기준값(평균 예측의 로그 오즈) {explainer.expected_value:.3f}")
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    i = int(np.argmax(m1.predict(te[FEATURES])))
    row = te.iloc[i]
    contrib = pd.Series(sv[i], index=FEATURES).sort_values(key=np.abs, ascending=False)
    p = 1 / (1 + np.exp(-(explainer.expected_value + sv[i].sum())))
    print(f"\n확률 1위 이벤트: 종목 {row.code} {row.date.date()}  예측 {p:.1%}  실제 연속 상한가 {int(row.y_close)}")
    for f in contrib.index[:6]:
        print(f"  {f:16s} 값 {row[f]:>10.4f}  기여 {contrib[f]:+.3f}")


if __name__ == "__main__":
    main()
