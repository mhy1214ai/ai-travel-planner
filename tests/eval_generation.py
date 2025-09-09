# tests/eval_generation.py
import sys
from pathlib import Path
from datetime import date, timedelta

# 프로젝트 루트 경로를 sys.path에 추가 (파일 직접 실행 대비)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa

def run_one(query: str):
    d0 = date.today() + timedelta(days=1)
    state = {
        "query": query,
        "city": "",  # region 파싱/추천은 상위 로직에서 처리된다고 가정
        "days": 3, "nights": 2, "ppl": 2, "with_kids": False,
        "start_date": d0.isoformat(), "end_date": (d0 + timedelta(days=2)).isoformat(),
        "themes": ["맛집", "전통시장"],
        "retrieved": [], "course_json": {}, "food_json": {}, "final_md": "",
    }
    g = build_graph()
    result = g.invoke(state)
    return result

CHECKS = [
    ("섹션 제목 포함(요약)", lambda md: ("일정 요약" in md) or ("요약" in md)),
    ("섹션 제목 포함(상세 일정)", lambda md: ("상세 일정" in md) or ("Day1" in md)),
    ("식당/먹거리 언급", lambda md: ("맛집" in md) or ("먹거리" in md)),
    ("날씨 기준 표기", lambda md: ("날씨:" in md) or ("기준:" in md)),
]

def main():
    queries = [
        "전주 한옥마을 중심 2박3일 코스",
        "부산 1박2일, 해운대 광안리 야경",
        "여수 2박3일, 비 오면 실내 대안 포함",
    ]
    for q in queries:
        res = run_one(q)
        md = res.get("final_md", "")
        print("\n=== ", q, " ===")
        for name, fn in CHECKS:
            print(f"[{'OK' if fn(md) else 'MISS'}] {name}")

if __name__ == "__main__":
    main()
