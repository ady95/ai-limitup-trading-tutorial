"""09-2 당일 첫 상한가 후보 탐지 — 등락률 순위에서 상한가 종목을 찾고 직전 60거래일 상한가 이력을 확인

    uv run python -m limitup.execution.candidates
"""

import pandas as pd

from limitup.collectors.kis import KIS
from limitup.events.detect import detect, live_end, load_daily

MIN_VALUE = 1_000_000_000


def limit_up_now(kis: KIS) -> pd.DataFrame:
    """KOSPI·KOSDAQ 등락률 상위 종목 중 현재가가 상한가인 종목"""
    rows = []
    body = kis.get("/uapi/domestic-stock/v1/ranking/fluctuation", "FHPST01700000", {
        "fid_cond_mrkt_div_code": "J", "fid_cond_scr_div_code": "20170", "fid_input_iscd": "0000",
        "fid_rank_sort_cls_code": "0", "fid_input_cnt_1": "0", "fid_prc_cls_code": "1",
        "fid_input_price_1": "", "fid_input_price_2": "", "fid_vol_cnt": "", "fid_trgt_cls_code": "0",
        "fid_trgt_exls_cls_code": "0", "fid_div_cls_code": "0", "fid_rsfl_rate1": "", "fid_rsfl_rate2": ""})
    for r in body["output"]:
        code = r["stck_shrn_iscd"]
        o = kis.get("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                    {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})["output"]
        rows.append({"code": code, "price": int(o["stck_prpr"]), "upper": int(o["stck_mxpr"]),
                     "base": int(o["stck_sdpr"]), "value": int(o["acml_tr_pbmn"]),
                     "chg": float(o["prdy_ctrt"])})
    df = pd.DataFrame(rows)
    return df[df.price == df.upper]


def first_limit_up(cands: pd.DataFrame) -> pd.DataFrame:
    """직전 60거래일 상한가 마감 이력이 없고, 분석 모집단(보통주)에 있는 종목만"""
    df = load_daily(end=live_end())         # 어제까지의 확정 일봉 (책 재현용 기준일이 아님)
    _, d = detect(df, 60)
    last = d.sort_values("date").groupby("code").tail(1).set_index("code")
    lu60 = d[d.date >= d.date.max() - pd.Timedelta(days=95)].groupby("code").lu_close.sum()
    out = cands[cands.code.isin(last.index)].copy()
    out["prior_lu_60d"] = out.code.map(lu60).fillna(0).astype(int)
    out["history_ok"] = out.code.map(last.history_ok).fillna(False)
    return out[(out.prior_lu_60d == 0) & out.history_ok & (out.value >= MIN_VALUE)]


if __name__ == "__main__":
    kis = KIS("real")
    now = limit_up_now(kis)
    print(f"현재 상한가 {len(now)}종목")
    print(now.to_string(index=False))
    f = first_limit_up(now).assign(n_limit_up=len(now))      # 당일 상한가 종목 수 (시장 피처)
    print(f"\n첫 상한가 후보 {len(f)}종목")
    print(f.to_string(index=False))
    f.to_csv("data/processed/today_candidates.csv", index=False)
