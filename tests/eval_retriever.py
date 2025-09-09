# tests/eval_retriever.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag import RAGPipeline

# 평가 엄격도: "loose"(권장) / "strict"
EVAL_MODE = "loose"

# 두 PDF(도시 감성·골목 맛집) 내용에 맞춘 현실적인 키워드 기준
QUERIES = [
    {
        "q": "서순라길·종묘·낙산 한 바퀴(3–4시간) + 전통 디저트/전통주 코스",
        "must_all": ["서순라길", "종묘"],
        "must_any": ["낙산", "한양도성", "청계천", "세운상가", "전통주", "과편", "가래떡", "막걸리"],
        "any_at_least": 2,
    },
    {
        "q": "선정릉 숲길 힐링 → 코엑스 별마당도서관 → 삼성동 카페 1일 루트",
        "must_all": ["선정릉", "별마당", "코엑스"],
        "must_any": ["삼성동", "정자", "숲길", "힐링", "디저트"],
        "any_at_least": 1,
    },
    {
        "q": "마곡 서울식물원 온실·습지원 + 한강전망데크 산책",
        "must_all": ["서울식물원", "온실"],
        "must_any": ["습지", "습지원", "한강전망데크", "마곡", "호수원"],
        "any_at_least": 1,
    },
    {
        "q": "419카페거리 브런치/디저트 + 솔밭근린공원 산책",
        "must_all": ["419", "카페거리"],
        "must_any": ["솔밭근린공원", "북한산", "수유"],
        "any_at_least": 1,
    },
    {
        "q": "용리단길 K-POP 스폿 + 국립중앙박물관 남산뷰 포토스팟 + 라자냐",
        "must_all": ["용리단길", "국립중앙박물관"],
        "must_any": ["남산", "포토", "하이브", "라자냐", "리소토"],
        "any_at_least": 1,
    },
]

def _normalize(s: str) -> str:
    return (s or "").lower()

def _joined_text(docs):
    return " ".join((d.get("content") or "") for d in docs)

def _check(doc_text: str, must_all, must_any, any_at_least) -> tuple[bool, list[str], list[str]]:
    t = _normalize(doc_text)
    hit_all = [k for k in must_all if _normalize(k) in t]
    hit_any = [k for k in must_any if _normalize(k) in t]
    if EVAL_MODE == "strict":
        ok = (len(hit_all) == len(must_all)) and (len(hit_any) >= any_at_least)
    else:
        # loose: all 중 최소 1개 이상 + any 기준 충족이면 OK
        ok = (len(must_all) == 0 or len(hit_all) >= 1) and (len(hit_any) >= any_at_least)
    return ok, hit_all, hit_any

def show_top(docs, topn=3):
    print("  Top results:")
    for i, d in enumerate(docs[:topn], 1):
        meta = d.get("meta", {}) or {}
        fn = meta.get("file_name") or meta.get("source")
        pg = meta.get("page")
        snippet = (d.get("content") or "").replace("\n", " ")[:140]
        print(f"   {i:>2}. {fn} [p.{pg}]  {snippet}...")

def main():
    rag = RAGPipeline(k=5)
    hit, total = 0, 0
    for item in QUERIES:
        q = item["q"]
        docs = rag.retrieve(q)
        text_blob = _joined_text(docs)
        ok, hit_all, hit_any = _check(
            text_blob,
            item.get("must_all", []),
            item.get("must_any", []),
            item.get("any_at_least", 0),
        )
        total += 1
        hit += 1 if ok else 0
        tag = "OK" if ok else "MISS"
        print(f"[{tag}] {q}  -> retrieved={len(docs)}  | hit_all={hit_all}  hit_any={hit_any}")
        show_top(docs, topn=3)
    print(f"HitRate@k={hit}/{total} = {hit/total:.2f}")

if __name__ == "__main__":
    main()
