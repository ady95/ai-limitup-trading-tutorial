"""AI로 첫 상한가 종목의 연속 상승 예측과 자동매매 따라하기 — 예제 코드 패키지"""

import sys

# Windows 콘솔(cp949)에서도 한글 출력이 깨지지 않게 한다
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
