# src/rag.py
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

from langchain_community.vectorstores import FAISS
from .config import get_embeddings

# BM25 (없어도 동작하도록 optional import)
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None

INDEX_DIR = Path("vectorstore/faiss_index")
DATA_DIR = Path("data")


# ======== 간단 키워드 추출/재랭킹 유틸 ========
def _extract_keywords_ko(q: str) -> List[str]:
    """아주 단순한 한국어 키워드 추출: 공백/구두점 분리 + 불용어 제거."""
    q = re.sub(r"[^\w가-힣\s]", " ", q)
    toks = [t for t in re.split(r"\s+", q) if t]
    stop = {
        "여행", "일정", "코스", "추천", "중심", "포함", "대안",
        "2박3일", "1박2일", "3일", "2일", "비오면", "비", "포함",
    }
    return [t for t in toks if t not in stop and len(t) >= 2]

def _keyword_score(text: str, kws: List[str]) -> int:
    text_low = (text or "").lower()
    return sum(text_low.count(k.lower()) for k in kws)

def _rerank_by_keywords(query: str, docs: List[Dict[str, Any]], boost: int = 2) -> List[Dict[str, Any]]:
    """벡터/하이브리드 상위 결과에 간단 키워드 점수를 더해 재랭킹."""
    kws = _extract_keywords_ko(query)
    if not kws:
        return docs
    scored = []
    for d in docs:
        txt = d.get("content") or ""
        s = _keyword_score(txt, kws) * boost
        scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


# ======== RRF(Reciprocal Rank Fusion) 병합 ========
def _doc_key(doc) -> str:
    """문서 고유키 생성(파일 경로/이름 + 페이지)."""
    meta = getattr(doc, "metadata", {}) or {}
    src = str(meta.get("source") or meta.get("file_name") or "")
    pg = str(meta.get("page") or "")
    return f"{src}#{pg}"

def _rrf_merge(doc_lists: List[List], k: int, c: int = 60) -> List:
    """
    여러 랭킹 리스트(FAISS, BM25 등)를 RRF로 병합.
    doc_lists: [[Document,...], [Document,...], ...]
    """
    scores = {}
    order = {}
    for li, docs in enumerate(doc_lists):
        for rank, d in enumerate(docs):
            key = _doc_key(d)
            # Reciprocal rank score
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            # 첫 등장 순서 기억(안정 정렬용)
            if key not in order:
                order[key] = (li, rank, d)
    # 점수로 정렬
    sorted_keys = sorted(scores.keys(), key=lambda k2: scores[k2], reverse=True)
    merged = [order[k2][2] for k2 in sorted_keys]
    return merged[:k]


class RAGPipeline:
    def __init__(self, k: int = 5, auto_ingest_if_missing: bool = True):
        self.vs = None
        self.k_default = k
        self.auto_ingest_if_missing = auto_ingest_if_missing
        self._bm25 = None  # BM25 retriever (optional)
        self._all_docs_cache = []  # for BM25 초기화용

    def _init_bm25_from_vs(self):
        """FAISS docstore에서 전체 문서를 꺼내 BM25 인덱스 구성."""
        if BM25Retriever is None or self.vs is None:
            return
        try:
            # LangChain FAISS는 docstore에 원본 Document를 들고 있음
            docs_all = list(self.vs.docstore._dict.values())
            if not docs_all:
                return
            self._all_docs_cache = docs_all
            self._bm25 = BM25Retriever.from_documents(docs_all)
            # k는 호출 시 지정하므로 여기서 고정하지 않음
            # (필요하면 self._bm25.k = self.k_default)
        except Exception:
            # BM25 실패해도 FAISS만으로 동작
            self._bm25 = None

    def _ensure_loaded(self) -> bool:
        if INDEX_DIR.exists():
            try:
                self.vs = FAISS.load_local(
                    str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
                )
                # BM25 인덱스 준비(1회)
                self._init_bm25_from_vs()
                return True
            except Exception:
                return False
        if self.auto_ingest_if_missing:
            try:
                from .ingest import build_faiss_from_folder
                build_faiss_from_folder(DATA_DIR, INDEX_DIR)
                self.vs = FAISS.load_local(
                    str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True
                )
                self._init_bm25_from_vs()
                return True
            except Exception:
                return False
        return False

    def load(self) -> bool:
        return self._ensure_loaded()

    def _faiss_search(self, query: str, k: int, fetch_k: int):
        """FAISS: MMR 우선, 실패 시 일반 검색."""
        try:
            return self.vs.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=0.5
            )
        except Exception:
            return self.vs.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int):
        """BM25: 있으면 사용, 없으면 빈 리스트."""
        if self._bm25 is None:
            return []
        # k를 조금 넉넉히 뽑아도 됨
        self._bm25.k = max(k, 10)
        try:
            return self._bm25.get_relevant_documents(query)
        except Exception:
            return []

    # ======== 하이브리드 검색 + RRF + 키워드 재랭킹 ========
    def retrieve(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        """
        1) FAISS(MMR)와 BM25를 각각 검색
        2) RRF로 병합하여 상호보완
        3) 간단 키워드 점수로 최종 재랭킹(지명/핵심어 우선)
        """
        if not self.vs and not self._ensure_loaded():
            return []

        topk = k or self.k_default
        fetch_k = max(40, topk * 6)

        faiss_docs = self._faiss_search(query, k=fetch_k, fetch_k=fetch_k)
        bm25_docs = self._bm25_search(query, k=fetch_k)

        # 병합: RRF (둘 중 하나가 비어도 안전)
        merged_docs = _rrf_merge([faiss_docs, bm25_docs], k=fetch_k)

        # dict로 변환
        out = [{"content": d.page_content, "meta": d.metadata} for d in merged_docs]

        # 키워드 재랭킹 후 topk
        out = _rerank_by_keywords(query, out, boost=2)
        return out[:topk]
        
    def suggest_queries(self, profile=None, topk=8, json_path="data/suggested_queries.json"):
        import json
        from pathlib import Path
        items = []
        p = Path(json_path)
        if p.exists():
            data = json.loads(p.read_text("utf-8"))

            # 1) 그룹형 스키마 지원: {"groups":[{"title":..., "items":[{"label","query"}]}]}
            if isinstance(data, dict) and isinstance(data.get("groups"), list):
                for g in data["groups"]:
                    for it in g.get("items", []):
                        label = (it.get("label") or it.get("query") or "").strip()
                        query = (it.get("query") or it.get("label") or "").strip()
                        if query:
                            items.append({"label": label or query, "query": query})

            # 2) 평면형 스키마 지원: [{"text":"...", "category":"..."} ...]
            elif isinstance(data, list):
                for it in data:
                    txt = (it.get("text") or "").strip()
                    if txt:
                        items.append({"label": txt, "query": txt})

        # 프로필 가중치(선택)
        if profile and profile.get("with_kids"):
            items.sort(key=lambda x: ("아이" in x["label"] or "가족" in x["label"]), reverse=True)

        # fallback
        if not items:
            items = [
                {"label":"전주 한옥마을 클래식 코스", "query":"전주 한옥마을 중심 1일 코스 + 전통시장 + 디저트"},
                {"label":"부산 해변/카페 루트", "query":"부산 해운대/광안리 바다 + 카페 2곳 1일 코스"},
                {"label":"강릉 카페 투어", "query":"강릉 안목해변 카페거리 + 포토스팟 반나절"},
                {"label":"여수 야경 루트", "query":"여수 돌산대교 야경 + 낭만포차 + 카페"},
            ]

        return items[:topk] if topk else items


    # def suggest_queries(
    #     self,
    #     profile: Dict[str, Any] | None = None,
    #     topk: int = 8,
    #     json_path: str | Path = "data/suggested_queries.json",
    # ) -> List[Dict[str, str]]:
    #     """
    #     추천 검색어를 반환.
    #     - 우선순위 1: data/suggested_queries.json (그룹/라벨/쿼리 구조)
    #     - 없으면 간단한 기본값 fallback
    #     반환 형식: [{"label": "...", "query": "..."}, ...]
    #     """
    #     # 1) JSON 기반
    #     try:
    #         p = Path(json_path)
    #         if p.exists():
    #             data = json.loads(p.read_text("utf-8"))
    #             groups = data.get("groups", [])
    #             items: List[Dict[str, str]] = []
    #             for g in groups:
    #                 for it in g.get("items", []):
    #                     lbl = str(it.get("label", "")).strip()
    #                     q = str(it.get("query", "")).strip()
    #                     if q:
    #                         # 라벨이 없으면 쿼리를 버튼 텍스트로 사용
    #                         items.append({"label": lbl or q, "query": q})
    #             # 프로필 기반 간단 필터 (아이동반이면 '아이/가족' 키워드 우선 정렬 등)
    #             if profile and profile.get("with_kids"):
    #                 items.sort(key=lambda x: ("아이" in x["label"] or "아이" in x["query"] or "가족" in x["label"]), reverse=True)
    #             return items[:topk] if topk else items
    #     except Exception:
    #         pass

    #     # 2) 간단 fallback (파일 없거나 파싱 실패 시)
    #     fallback = [
    #         {"label": "전주 한옥마을 클래식 코스", "query": "전주 한옥마을 중심 1일 코스 + 전통시장 + 디저트"},
    #         {"label": "부산 해변/카페 루트", "query": "부산 해운대/광안리 바다 + 카페 2곳 1일 코스"},
    #         {"label": "강릉 카페 투어", "query": "강릉 안목해변 카페거리 + 포토스팟 반나절"},
    #         {"label": "여수 야경 루트", "query": "여수 돌산대교 야경 + 낭만포차 + 카페"},
    #         {"label": "속초 설악산/바다", "query": "속초 설악산 가벼운 트레킹 + 해변 + 맛집"},
    #     ]
    #     if profile and profile.get("with_kids"):
    #         fallback.insert(0, {"label": "아이동반 실내 코스(서울)", "query": "서울 국립중앙박물관 + 실내 포토스팟 + 카페"})
    #     return fallback[:topk] if topk else fallback
    