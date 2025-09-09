# src/ingest.py
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import TextLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader  # 우선
except Exception:
    PyMuPDFLoader = None
try:
    from langchain_community.document_loaders import PyPDFLoader    # 폴백
except Exception:
    PyPDFLoader = None

from src.config import CHUNK_SIZE, CHUNK_OVERLAP  # ← .env 반영

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .config import get_embeddings


# 기본 경로 (프로젝트 컨벤션)
DEFAULT_DATA_DIR = Path("data")
DEFAULT_INDEX_DIR = Path("vectorstore") / "faiss_index"
DEFAULT_META_PKL = Path("vectorstore") / "docs_meta.pkl"


def _load_single_file(path: Path) -> List:
    ext = path.suffix.lower()
    if ext == ".pdf":
        if PyMuPDFLoader:
            return PyMuPDFLoader(str(path)).load()
        if PyPDFLoader:
            return PyPDFLoader(str(path)).load()
        raise RuntimeError("PDF 파서(pymupdf/pypdf) 설치 필요")
    elif ext in {".txt", ".md", ".markdown"}:
        loader = TextLoader(str(path), autodetect_encoding=True)
        return loader.load()
    return []


def _load_all_documents(data_dir: Path) -> List:
    """data_dir 아래 지원되는 모든 파일을 로드하여 Documents 리스트 반환."""
    if not data_dir.exists():
        raise FileNotFoundError(f"data 폴더가 없습니다: {data_dir.resolve()}")

    docs: List = []
    # 지원 확장자
    patterns = ["**/*.pdf", "**/*.txt", "**/*.md", "**/*.markdown"]
    for pat in patterns:
        for p in data_dir.glob(pat):
            try:
                file_docs = _load_single_file(p)
                docs.extend(file_docs)
            except Exception as e:
                print(f"[ingest][WARN] 파일 로드 실패: {p} -> {e}")

    if not docs:
        raise ValueError(f"[ingest] 로드 가능한 문서를 찾지 못했습니다. 폴더: {data_dir.resolve()}")
    return docs


def _split_documents(docs: List) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # ← .env 값 사용
        chunk_overlap=CHUNK_OVERLAP, # ← .env 값 사용
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def _texts_and_metas_from_docs(docs: List) -> Tuple[List[str], List[Dict[str, Any]]]:
    """LC Documents -> (texts, metadatas)"""
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        meta = dict(d.metadata or {})
        # 메타 정리: source/file/page 등 표준화
        src = meta.get("source") or meta.get("file_path") or meta.get("path")
        if not src and "pdf" in str(meta).lower():
            # 일부 로더는 'source'를 안 넣기도 함
            src = meta.get("pdf_path")
        if not src:
            # 최후 방안: title이나 기타에서 유추
            src = meta.get("title") or "unknown_source"
        meta["source"] = str(src)

        # page 번호 정규화
        if "page" in meta and isinstance(meta["page"], (int, str)):
            try:
                meta["page"] = int(meta["page"])
            except Exception:
                pass

        # 파일명/확장자 보강
        try:
            p = Path(meta["source"])
            meta.setdefault("file_name", p.name)
            meta.setdefault("ext", p.suffix.lower())
            meta.setdefault("dir", str(p.parent))
        except Exception:
            pass

        texts.append(text)
        metas.append(meta)
    return texts, metas


def build_faiss_from_folder(
    data_dir: Path = DEFAULT_DATA_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    meta_pkl: Path = DEFAULT_META_PKL,
) -> Tuple[FAISS, List[str]]:
    """
    data_dir의 문서를 로드 → 청킹 → 임베딩 → FAISS 인덱스 생성 & 저장.
    또한 vectorstore/docs_meta.pkl에 문서 메타를 저장(테마 추출 등에 활용).

    Returns:
        (vs, texts)
    """
    data_dir = Path(data_dir)
    index_dir = Path(index_dir)
    meta_pkl = Path(meta_pkl)

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    meta_pkl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Loading documents from: {data_dir.resolve()}")
    raw_docs = _load_all_documents(data_dir)

    print(f"[ingest] Splitting documents (count={len(raw_docs)}) ...")
    docs = _split_documents(raw_docs)

    print(f"[ingest] Building texts & metadatas (chunks={len(docs)}) ...")
    texts, metas = _texts_and_metas_from_docs(docs)

    if not texts:
        raise ValueError("[ingest] texts 가 비었습니다. 로드/청킹 결과를 확인하세요.")

    print(f"[ingest] Initializing embeddings ...")
    embeddings = get_embeddings()

    print(f"[ingest] Creating FAISS index (chunks={len(texts)}) ...")
    # ✅ 핵심: 'embedding' 인자명 (단수)
    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metas)

    print(f"[ingest] Saving index to: {index_dir.resolve()}")
    vs.save_local(str(index_dir))

    # docs_meta.pkl 작성: 파일/페이지 수준 메타 요약
    # (load_theme_options 등에서 파일명 기반 테마 추출에 사용)
    try:
        print(f"[ingest] Writing meta pickle: {meta_pkl.resolve()}")
        # 원본 파일 기반 메타만 추려서 저장(너무 큰 텍스트는 제외)
        file_level_meta: List[Dict[str, Any]] = []
        for m in metas:
            file_level_meta.append(
                {
                    "source": m.get("source"),
                    "file_name": m.get("file_name"),
                    "ext": m.get("ext"),
                    "page": m.get("page"),
                }
            )
        with open(meta_pkl, "wb") as f:
            pickle.dump(file_level_meta, f)
    except Exception as e:
        print(f"[ingest][WARN] docs_meta.pkl 저장 실패: {e}")

    print("[ingest] Done.")
    return vs, texts

# --- 추천 검색어 자동 생성 (PDF 전체 기반) ---
import re, json
from collections import Counter
from pathlib import Path

KO_STOPWORDS = {"여행","일정","코스","추천","가이드","대한민국","소개","안내","관광"}
SUFFIX_HINTS = ("구","군","시","동","읍","면","로","길","산","공원","해변","해수욕장","시장","박물관","미술관","카페","거리","한옥마을","전망대")

def _extract_phrases(text: str) -> list[str]:
    toks = re.findall(r"[가-힣A-Za-z0-9·\-]{2,}", text or "")
    # 아주 단순한 후보(유니그램 + 접미사 가점)
    c = Counter(toks)
    scored = [(w, c[w] + (1 if any(w.endswith(s) for s in SUFFIX_HINTS) else 0)) for w in c]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:80]]

def _category_for(w: str) -> str:
    if w.endswith(("구","군","시","동","읍","면")): return "지역"
    if any(w.endswith(s) for s in ["박물관","미술관","시장","사찰","한옥마을","공원","해변","전망대"]): return "장소"
    return "테마"

def build_suggested_queries_from_pdfs(
    data_dir: Path = DEFAULT_DATA_DIR,
    out_path: Path = Path("data") / "suggested_queries.json",
    per_doc_topk: int = 12,
    global_topk: int = 60,
) -> None:
    """data/ 내 모든 PDF/텍스트를 훑어 label(짧음)과 query(조금 더 구체적)를 만든 뒤 JSON(평면형)으로 저장."""
    raw_docs = _load_all_documents(Path(data_dir))
    # 문서별 후보
    doc_keys = []
    for d in raw_docs:
        txt = (d.page_content or "").strip()
        if not txt: 
            continue
        doc_keys.extend(_extract_phrases(txt))

    if not doc_keys:
        out_path.write_text("[]", "utf-8"); return

    # 전역 상위 추출
    allc = Counter(doc_keys)
    picked = [w for w, _ in allc.most_common(global_topk)]

    out = []
    for w in picked:
        cat = _category_for(w)
        label = w  # 버튼에 짧게 표시
        if cat == "지역":
            query = f"{w} 여행 1일 코스"
        elif cat == "장소":
            query = f"{w} 중심 3~4시간 루트"
        else:
            query = f"{w} 체험 + 카페 2곳"
        out.append({"text": label, "category": cat, "query": query})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")


# 편의 함수: 기본 경로에 대해 인덱스 재생성
def rebuild_default_index() -> None:
    build_faiss_from_folder(DEFAULT_DATA_DIR, DEFAULT_INDEX_DIR, DEFAULT_META_PKL)


if __name__ == "__main__":
    # 터미널에서 바로 실행 시:
    #   python -m src.ingest
    # 또는
    #   python src/ingest.py
    rebuild_default_index()
