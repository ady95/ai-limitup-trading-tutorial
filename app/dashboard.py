"""10-2 Streamlit 모니터링 대시보드

    uv run streamlit run app/dashboard.py

탭: 개요 / 오늘의 후보 / 모의투자 체결 / 백테스트. 모든 숫자는 data/processed 의 결과 파일에서 읽는다.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from limitup.paths import PROCESSED

SERIES = "#2a78d6"            # 단일 계열 색 (기준 팔레트 1번)
GRID = "rgba(128,128,128,0.15)"
st.set_page_config(page_title="첫 상한가 모니터링", layout="wide")


def style(fig, y_pct=True):
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor="rgba(128,128,128,0.4)",
                     tickformat=".0%" if y_pct else None)
    return fig


@st.cache_data
def load_csv(name):
    p = PROCESSED / name
    return pd.read_csv(p, dtype={"code": str}) if p.exists() else pd.DataFrame()


@st.cache_data
def backtest_b():
    """B-마감(항상 체결 가정, 비용·슬리피지 0.1%)의 평가 기간 거래 — 08-5와 같은 계산"""
    from limitup.backtest.costs import net_return
    from limitup.backtest.run import data
    from limitup.strategies.rules import load_config, select
    test, _ = data()
    b = select(test, load_config(), "B").copy()
    b["net"] = [net_return(g, d, slip_sell=0.001) for g, d in zip(b.t1_open / b.t1_base - 1, b.t1_date)]
    b["half"] = b.date.dt.year.astype(str) + np.where(b.date.dt.month <= 6, " 상반기", " 하반기")
    return b[["date", "code", "pred_gap", "net", "half"]]


tab0, tab1, tab2, tab3 = st.tabs(["개요", "오늘의 후보", "모의투자 체결", "백테스트"])

fills = load_csv("paper_fill_log.csv")
with tab0:
    st.subheader("연구 상태")
    c = st.columns(4)
    events = pd.read_parquet(PROCESSED / "events.parquet")
    c[0].metric("첫 상한가 이벤트 (2021~)", f"{(events.group == 'close').sum():,}건")
    pred = pd.read_parquet(PROCESSED / "models" / "pred_test.parquet")
    from sklearn.metrics import average_precision_score
    c[1].metric("M1 평가 기간 PR-AUC", f"{average_precision_score(pred.y_close, pred.m1):.3f}",
                f"M0 대비 {average_precision_score(pred.y_close, pred.m1) - average_precision_score(pred.y_close, pred.m0):+.3f}")
    b = backtest_b()
    c[2].metric("전략 B 거래당 평균 (항상 체결 가정)", f"{b.net.mean():+.2%}")
    if len(fills):
        amt = (fills.filled_qty * fills.price).sum() / (fills.order_qty * fills.price).sum()
        c[3].metric("모의투자 금액 체결률", f"{amt:.1%}")
        c[3].caption(f"{fills.date.nunique()}거래일, 주문 {len(fills)}건 기준")
    st.caption("전략 B의 평균은 장 마감 단일가에 항상 체결된다는 가정의 값입니다. 실제 체결률은 '모의투자 체결' 탭을 보세요.")

with tab1:
    st.subheader("오늘의 첫 상한가 후보와 다음 날 시가 갭 예측")
    cand = load_csv("today_candidates_scored.csv")
    if len(cand) and "date" in cand and (cand.date.astype(str) == dt.date.today().isoformat()).all():
        st.dataframe(cand, hide_index=True, use_container_width=True)
    elif len(cand) and "date" in cand:
        st.info(f"오늘의 후보가 아직 없습니다 (daily close 실행 시 15:10 이후 생성). 아래는 {cand.date.iloc[0]}의 후보입니다.")
        st.dataframe(cand, hide_index=True, use_container_width=True)
    else:
        st.info("아직 오늘의 후보가 없습니다 (daily close 실행 시 15:10 이후 생성).")

with tab2:
    st.subheader("장 마감 단일가 상한가 매수의 체결 기록")
    if fills.empty:
        st.info("기록이 없습니다. daily close 를 실행하면 쌓입니다 (09-4).")
    else:
        f = fills.copy()
        f["auction_share"] = f.auction_volume / (f.auction_volume + f.bid_qty1_after_close)
        fig = px.scatter(f, x="auction_share", y="fill_ratio", hover_data=["date", "code", "order_qty", "filled_qty"],
                         title="대기열 대용 비율(종가 단일가 거래량 ÷ (거래량 + 15:31 매수 1호가 잔량)) 대 내 주문의 체결 비율",
                         labels={"auction_share": "대기열 대용 비율", "fill_ratio": "내 주문 체결 비율"})
        fig.update_traces(marker=dict(size=10, color=SERIES, line=dict(width=2, color="white")))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(128,128,128,0.5)", dash="dot"))
        top = max(0.3, f.auction_share.max() * 1.1, f.fill_ratio.max() * 1.1)
        fig.update_xaxes(tickformat=".0%", range=[0, top])
        fig = style(fig)
        fig.update_yaxes(range=[0, top])                    # 두 축을 같은 범위로 — 점선이 45도가 되게
        st.plotly_chart(fig, use_container_width=True)
        st.caption("점선은 내 주문 체결 비율이 대기열 대용 비율과 같은 경우입니다. 대용 비율은 서로 다른 시각의 거래량과 잔량으로 만든 값이라 실제 대기열 체결률이나 내 순번을 뜻하지 않습니다 (09-4).")
        pct = st.column_config.NumberColumn(format="percent")
        st.dataframe(f, hide_index=True, use_container_width=True,
                     column_config={"fill_ratio": pct, "auction_share": pct, "auction_share_of_queue": pct})

with tab3:
    st.subheader("전략 B 평가 기간 성과 (항상 체결 가정, 비용·슬리피지 0.1% 포함)")
    b = backtest_b()
    h = b.groupby("half").net.agg(mean="mean", n="size", win=lambda x: (x > 0).mean()).reset_index()
    fig = px.bar(h, x="half", y="mean", hover_data={"n": True, "win": ":.1%"}, title="반기별 거래당 평균 수익률",
                 labels={"half": "", "mean": "거래당 평균"})
    fig.update_traces(marker=dict(color=SERIES, cornerradius=4), width=0.35)
    st.plotly_chart(style(fig), use_container_width=True)
    st.dataframe(h.rename(columns={"half": "반기", "mean": "거래당 평균", "n": "거래 수", "win": "승률"}),
                 hide_index=True, use_container_width=True,
                 column_config={"거래당 평균": st.column_config.NumberColumn(format="percent"),
                                "승률": st.column_config.NumberColumn(format="percent")})
    st.warning("B-마감 사후 참고 성과입니다. 최종 마감 종목·확정 일봉 피처를 사후적으로 쓰고 전량 체결을 가정한 계산으로, 실행 가능한 전략의 성과도 실제 성과의 상한도 아닙니다 (07-3, 08-3).")
