"""02-2 첫 시세 조회 — 현재가·기준가·상한가·하한가

    uv run python -m limitup.collectors.quote 005930 035720
"""

import argparse

from limitup.collectors.kis import KIS
from limitup.events.price_limit import upper_limit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+")
    args = ap.parse_args()

    kis = KIS("real")
    print(f"{'종목':>6} {'현재가':>10} {'기준가':>10} {'상한가':>10} {'하한가':>10} {'계산 상한가':>10}")
    for code in args.codes:
        o = kis.get("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                    {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})["output"]
        base = int(o["stck_sdpr"])
        print(f"{code:>6} {int(o['stck_prpr']):>10,} {base:>10,} {int(o['stck_mxpr']):>10,} "
              f"{int(o['stck_llam']):>10,} {upper_limit(base):>10,}")


if __name__ == "__main__":
    main()
