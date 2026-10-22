"""06-4 AI 분석용 1분봉 차트 생성

    uv run python -m limitup.llm.chart          # 장중 피처가 있는 첫 상한가 이벤트의 T일 차트

사람이 보는 차트와 다른 점
- 축을 고정한다: 가로 = 장 시작부터 경과 분(0~390), 세로 = 기준가 대비 등락률(-30%~+32%)
  → 종목·날짜가 달라도 같은 위치가 같은 의미가 된다
- 기준선(0%)과 상한가 선을 그린다 → '상한가에 붙어 있는가'를 그림만 보고 판단할 수 있게
- 종목명·날짜·가격 숫자를 넣지 않는다 → 모델이 사전 지식으로 종목을 알아보지 못하게
- 보조지표·격자·색상 장식을 빼고 크기를 통일한다 (640 x 400)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from limitup.collectors.minute import load_minute
from limitup.features.intraday import minutes_since_open
from limitup.paths import PROCESSED

CHART_DIR = PROCESSED / "charts"


def draw(m: pd.DataFrame, base: int, upper: int, path) -> None:
    m = m.sort_values("time")
    x = minutes_since_open(m.time)
    ret = m.close / base - 1
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(6.4, 4.0), dpi=100, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, ret, color="black", linewidth=1.2)
    ax.axhline(upper / base - 1, color="red", linestyle="--", linewidth=1)     # 상한가
    ax.axhline(0, color="gray", linewidth=0.8)                                 # 기준가
    ax.set_ylim(-0.30, 0.32)
    ax.set_yticks([-0.3, 0, 0.3], ["-30%", "0%", "+30%"])
    axv.bar(x, m.volume / m.volume.max(), width=1.0, color="gray")
    axv.set_yticks([])
    axv.set_xlim(0, 390)
    axv.set_xticks([0, 120, 240, 390], ["open", "+2h", "+4h", "close"])
    for a in (ax, axv):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ds = pd.read_parquet(PROCESSED / "features.parquet")
    ev = ds[ds.has_intraday]
    daily = {}
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in ev.itertuples():
        path = CHART_DIR / f"{r.code}_{r.date:%Y%m%d}.png"
        if path.exists():
            continue
        if r.code not in daily:
            daily[r.code] = pd.read_parquet(f"data/raw/daily/{r.code}.parquet").set_index("date").base
        draw(load_minute(r.code, f"{r.date:%Y%m%d}"), int(daily[r.code].loc[r.date]), int(r.upper), path)
        n += 1
    print(f"차트 {n:,}장 생성 → {CHART_DIR}")


if __name__ == "__main__":
    main()
