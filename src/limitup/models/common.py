"""05장 공통: 데이터셋 불러오기, 피처 목록, 평가 지표, 날짜 단위 부트스트랩"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from limitup.features.daily import DAILY_FEATURES
from limitup.features.intraday import INTRADAY_FEATURES
from limitup.features.market import MARKET_FEATURES
from limitup.paths import PROCESSED

FEATURES = DAILY_FEATURES + MARKET_FEATURES          # M1 기본 피처 21개 (전체 기간에 있음)
ALL_FEATURES = FEATURES + INTRADAY_FEATURES          # 장중 피처 포함 26개 (최근 1년)
MODEL_DIR = PROCESSED / "models"


def load() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "features.parquet")


def parts(ds: pd.DataFrame):
    """학습(퍼지 제외)·검증(퍼지 제외)·평가 구간"""
    tr = ds[(ds.split == "train") & ~ds.purge]
    va = ds[(ds.split == "valid") & ~ds.purge]
    te = ds[ds.split == "test"]
    return tr, va, te


def precision_at(y, p, frac=0.1) -> float:
    """예측 확률 상위 frac 안의 실제 정답 비율"""
    k = max(1, int(round(len(p) * frac)))
    top = np.argsort(-np.asarray(p))[:k]
    return float(np.asarray(y)[top].mean())


def daily_top1(df: pd.DataFrame, score: str, y: str = "y_close") -> float:
    """이벤트가 있는 날마다 점수 1위 한 종목만 골랐을 때의 적중률"""
    top = df.loc[df.groupby("date")[score].idxmax()]
    return float(top[y].mean())


def metrics(df: pd.DataFrame, score: str, y: str = "y_close") -> dict:
    yt, p = df[y].to_numpy(), df[score].to_numpy()
    return {"n": len(df), "base": yt.mean(), "pr_auc": average_precision_score(yt, p),
            "roc_auc": roc_auc_score(yt, p), "brier": brier_score_loss(yt, np.clip(p, 0, 1)),
            "p_at_10": precision_at(yt, p, 0.1), "top1_day": daily_top1(df, score, y)}


def cluster_boot(df: pd.DataFrame, fn, n_boot=1000, seed=0):
    """날짜 단위로 재표본추출해 fn(df) 의 95% 신뢰구간을 구한다 (03-3과 같은 이유)"""
    rng = np.random.default_rng(seed)
    days = df.date.unique()
    groups = {d: g for d, g in df.groupby("date")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        b = pd.concat([groups[d] for d in pick], ignore_index=True)
        if b.y_close.nunique() == 2:
            vals.append(fn(b))
    return np.percentile(vals, [2.5, 97.5])


def fmt(m: dict) -> str:
    return (f"n={m['n']:>5}  기준선 {m['base']:.1%}  PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
            f"Brier {m['brier']:.4f}  상위10% 적중 {m['p_at_10']:.1%}  하루1위 적중 {m['top1_day']:.1%}")
