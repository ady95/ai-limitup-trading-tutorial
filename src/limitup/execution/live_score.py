"""09-3 장중 정보로 오늘 후보의 점수(다음 날 시가 갭 예측)를 계산한다

    uv run python -m limitup.execution.live_score data/processed/today_candidates.csv

장 마감 단일가 직전(15:10~15:19)의 현재가·거래대금을 '오늘의 일봉'처럼 붙여 4장의 피처를 다시 만든다.
- 일봉 피처: 과거 일봉 + 오늘 스냅숏 (종가 = 현재가, 거래대금 = 누적 거래대금)
- 시장 피처: 당일 상한가 종목 수(등락률 순위), 지수 등락·상승 종목 비율(업종 현재지수 API)
- 업종 피처: 장중에 전 종목을 볼 수 없으므로 0으로 둔다 (05-5에서 기여가 거의 없던 피처)
모델은 지금까지의 모든 첫 상한가로 학습한다 (실시간 판단에는 최신 데이터가 가장 유리, 05-3).
"""

import argparse
import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd

from limitup.collectors.kis import KIS
from limitup.events.detect import load_daily
from limitup.features.daily import stock_features
from limitup.models.common import FEATURES, load
from limitup.models.regress import BASE
from limitup.paths import INDEX_FILE

INDEX_CODES = {"KOSPI": "0001", "KOSDAQ": "1001"}


def snapshot(kis, code):
    o = kis.get("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})["output"]
    return {"open": int(o["stck_oprc"]), "high": int(o["stck_hgpr"]), "low": int(o["stck_lwpr"]),
            "close": int(o["stck_prpr"]), "volume": int(o["acml_vol"]), "value": int(o["acml_tr_pbmn"]),
            "base": int(o["stck_sdpr"]), "upper": int(o["stck_mxpr"])}


def index_now(kis, market):
    o = kis.get("/uapi/domestic-stock/v1/quotations/inquire-index-price", "FHPUP02100000",
                {"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": INDEX_CODES[market]})["output"]
    up, down, flat = int(o["ascn_issu_cnt"]), int(o["down_issu_cnt"]), int(o["stnr_issu_cnt"])
    return {"close": float(o["bstp_nmix_prpr"]), "change_rate": float(o["bstp_nmix_prdy_ctrt"]) / 100,
            "breadth": up / max(up + down + flat, 1)}


def build_rows(cands, kis, today):
    df = load_daily()
    hist_lu = df[~df.halted].groupby("date").lu_close.sum()
    ix = pd.read_parquet(INDEX_FILE).sort_values("date")
    idx = {m: index_now(kis, m) for m in INDEX_CODES}
    n_lu = int(cands.n_limit_up.iloc[0]) if "n_limit_up" in cands else len(cands)
    rows = []
    for c in cands.itertuples():
        s = snapshot(kis, c.code)
        hist = df[df.code == c.code]
        market = hist.market.iloc[-1]
        today_row = {**{k: hist.iloc[-1][k] for k in hist.columns}, **s, "date": pd.Timestamp(today),
                     "change": s["close"] - s["base"], "halted": False,
                     "lu_touch": s["high"] >= s["upper"], "lu_close": s["close"] == s["upper"]}
        f = stock_features(pd.concat([hist, pd.DataFrame([today_row])], ignore_index=True)).iloc[-1]
        prev5 = ix[ix["index"] == market].close.iloc[-5]
        row = {"code": c.code, **{k: f[k] for k in FEATURES if k.startswith("f_")},
               "m_n_lu": n_lu, "m_lu_heat": n_lu / hist_lu.iloc[-20:].mean(), "m_breadth": idx[market]["breadth"],
               "m_idx_ret": idx[market]["change_rate"], "m_idx_ret5": idx[market]["close"] / prev5 - 1,
               "s_ind_strong": 0, "s_ind_lu": 0, "s_leader": 1}
        rows.append(row)
    return pd.DataFrame(rows)


def gap_model():
    ds = load()
    return lgb.train({**BASE, "objective": "huber", "alpha": 0.05}, lgb.Dataset(ds[FEATURES], ds.y_ret_open),
                     num_boost_round=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    a = ap.parse_args()
    cands = pd.read_csv(a.csv, dtype={"code": str})
    kis = KIS("real")
    rows = build_rows(cands, kis, dt.date.today())
    rows["pred_gap"] = gap_model().predict(rows[FEATURES])
    rows = rows.sort_values("pred_gap", ascending=False)
    print(rows[["code", "pred_gap", "f_value_log", "f_gap", "m_n_lu", "m_breadth", "m_idx_ret"]].to_string(index=False))
    rows.to_csv(a.csv.replace(".csv", "_scored.csv"), index=False)


if __name__ == "__main__":
    main()
