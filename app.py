# --- app.py (맨 위) ---
import os, json, re, pickle, sqlite3, uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# set_page_config 이후에만 내부 모듈 import
from src.graph import build_graph
from src.rag import RAGPipeline
from src.ingest import build_faiss_from_folder
from src.config import get_langfuse, get_llm

# ⬇️ 반드시 첫 Streamlit 명령이어야 함 (+ 중복 방지 가드)
if not st.session_state.get("_pagecfg_done"):
    st.set_page_config(page_title="AI 여행 플래너", page_icon="🧭", layout="wide")
    st.session_state["_pagecfg_done"] = True

# --- 추천 검색어 세션키 보증 ---
if "suggest_keywords" not in st.session_state:
    st.session_state["suggest_keywords"] = []
if "last_clicked_suggest" not in st.session_state:
    st.session_state["last_clicked_suggest"] = ""
if "run_requested" not in st.session_state:
    st.session_state["run_requested"] = False

st.title("🧭 AI 여행 플래너")

PROFILE_PATH = "memory/user_profile.json"
DB_PATH = os.getenv("DB_PATH", "./memory/history.db")
os.makedirs("memory", exist_ok=True)

# ---------- 유틸 ----------
def load_json(p, default):
    if os.path.exists(p):
        return json.load(open(p, "r", encoding="utf-8"))
    return default

def save_json(p, obj):
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def default_profile():
    tmr = date.today() + timedelta(days=1)
    return {
        "ppl": 2,
        "arrival_date": tmr.isoformat(),
        "depart_date": (tmr + timedelta(days=2)).isoformat(),  # 2박3일
        "with_kids": False,
        "dietary": [],
        "interests": ["food", "culture"],
        "focus": "한옥·전통시장",
        "focus_list": [],
        "city": "서울",
    }

def parse_region_from_query(q: str) -> str:
    if not q:
        return ""
    m = re.search(r"(.+?)\s*여행", q)
    if m:
        region = m.group(1).strip()
        return region.split()[0] if len(region) > 20 else region
    toks = re.split(r"\s+", q.strip())
    if toks and any(x.endswith("월") for x in toks):
        return ""
    return toks[0] if toks else ""

def to_date(v, fallback: date) -> date:
    if isinstance(v, date): return v
    if isinstance(v, str) and v:
        try: return datetime.strptime(v, "%Y-%m-%d").date()
        except Exception: return fallback
    return fallback

def compute_days(a: date, d: date):
    delta = (d - a).days
    if delta < 1:
        return None, "종료일은 시작일 이후여야 합니다."
    nights = delta
    days = nights + 1
    return (days, nights), None

def load_theme_options() -> list[str]:
    try:
        tj = os.path.join("data", "themes.json")
        if os.path.exists(tj):
            arr = json.load(open(tj, "r", encoding="utf-8"))
            if isinstance(arr, list):
                return sorted({str(x).strip() for x in arr if str(x).strip()})
    except Exception:
        pass
    try:
        meta_p = os.path.join("vectorstore", "docs_meta.pkl")
        if os.path.exists(meta_p):
            metas = pickle.load(open(meta_p, "rb"))
            cands = []
            for m in metas:
                base = os.path.basename(str(m.get("source","")))
                text = re.sub(r"[^가-힣\s]", " ", base)
                cands.extend([t for t in text.split() if len(t) >= 2])
            stop = {"가이드","가이드북","지도","관광","여행","대한민국","국문","영문","코스","소개","안내"}
            uniq = sorted({c for c in cands if c not in stop})
            if uniq: return uniq[:50]
    except Exception:
        pass
    return ["해변","한옥","전통시장","미술관","박물관","카페거리","야시장","온천","등산","유람선","서핑","수목원","벚꽃","단풍","야경","맛집","호캉스","체험마을"]

def extract_month_from_text(q: str) -> int | None:
    if not q: return None
    m = re.search(r"(\d{1,2})\s*월", q)
    if m:
        mm = int(m.group(1))
        if 1 <= mm <= 12: return mm
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
    tok = re.findall(r"[A-Za-z]{3,}", q.lower())
    for t in tok:
        if t in month_map: return month_map[t]
    return None

def llm_suggest_regions(query: str, month: int | None, with_kids: bool, themes: list[str]) -> list[dict]:
    """추천 지역 LLM 호출 (없으면 안전한 fallback)"""
    try:
        llm = get_llm()
    except Exception:
        llm = None
    prompt = (
        "당신은 한국 여행 플래너입니다. 사용자가 월/조건만 말해도, 한국 내 여행지 5곳을 추천하세요.\n"
        "- 지역명은 간결하게(예: 부산, 전주, 강릉, 속초 등)\n"
        "- 추천 사유 1~2문장\n"
        "- 결과는 JSON 배열만, 각 항목은 {\"region\":\"도시\",\"why\":\"사유\"}\n\n"
        f"사용자 질의: {query}\n월: {month if month else '불명'}\n아이 동반: {with_kids}\n테마: {', '.join(themes) if themes else '없음'}\n"
    )
    fallback = [
        {"region":"강릉","why":"가을 바다와 카페 투어가 좋아요."},
        {"region":"전주","why":"한옥마을과 전통 먹거리가 풍부해요."},
        {"region":"부산","why":"해변과 도심 먹거리 모두 즐기기 좋아요."},
        {"region":"여수","why":"야경과 해산물, 섬 투어가 좋아요."},
        {"region":"속초","why":"설악산 단풍과 바다를 함께 즐겨요."},
    ]
    if not llm:
        return fallback
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"(\[.*\])", text, re.S)
        raw = m.group(1) if m else text
        data = json.loads(raw)
        recs = []
        for it in data:
            region = str(it.get("region","")).strip()
            why = str(it.get("why","")).strip()
            if region:
                recs.append({"region": region, "why": why})
        return recs[:8] or fallback
    except Exception:
        return fallback

# --- 광역 지역 처리 ---
BROAD_REGIONS = {
    "서해","수도권","영남","호남","충청","강원권","전라도","경상도","충청도"
}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def is_broad_region(name: str) -> bool:
    return _norm(name) in {_norm(x) for x in BROAD_REGIONS}

def llm_expand_broad_region(broad: str, month: int | None, with_kids: bool, themes: list[str]) -> list[dict]:
    """
    '서해/영남/호남' 같은 광역 키워드 → 하위 도시 후보를 LLM으로 제안. 실패 시 fallback.
    """
    try:
        llm = get_llm()
    except Exception:
        llm = None

    prompt = (
        f"사용자 요청 지역이 '{broad}'처럼 광역 범위입니다. 한국의 구체 도시/지역 5~8개를 추천하세요.\n"
        "- 예: 서해 → 태안, 보령, 군산, 목포, 신안, 강화도...\n"
        "- 각 항목은 JSON: {{\"region\":\"도시/군/구\",\"why\":\"간단한 사유\"}}\n"
        f"- 월: {month if month else '불명'}, 아이동반: {with_kids}, 테마: {', '.join(themes) if themes else '없음'}\n"
        "JSON 배열만 출력하세요."
    )
    fallback_map = {
        "서해": [
            {"region": "태안", "why": "해변·수목원·우천 대안 있음"},
            {"region": "보령", "why": "대천해수욕장·머드박물관"},
            {"region": "군산", "why": "근대역사·근대미술관·실내전시"},
            {"region": "목포", "why": "해양문화·도시관광·실내공간 다수"},
            {"region": "강화도", "why": "근교·사찰·마니산·우천 대안"},
        ],
    }
    if not llm:
        return fallback_map.get(broad, [])

    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        m = re.search(r"(\[.*\])", text, re.S)
        raw = m.group(1) if m else text
        data = json.loads(raw)
        out = []
        for it in data:
            r = str(it.get("region", "")).strip()
            why = str(it.get("why", "")).strip()
            if r:
                out.append({"region": r, "why": why})
        return out[:8] or fallback_map.get(broad, [])
    except Exception:
        return fallback_map.get(broad, [])

# --- 추천어 JSON 로딩 유틸 ---
def load_suggest_keywords(path: str = "data/suggested_queries.json", topk: int = 12):
    try:
        if not os.path.exists(path):
            return []
        data = json.load(open(path, "r", encoding="utf-8"))
        items = []
        # 평면형/그룹형 모두 지원
        if isinstance(data, list):
            for it in data:
                label = (it.get("text") or it.get("label") or "").strip()
                query = (it.get("query") or label).strip()
                if label:
                    items.append({"label": label, "query": query})
        elif isinstance(data, dict) and isinstance(data.get("groups"), list):
            for g in data["groups"]:
                for it in g.get("items", []):
                    label = (it.get("label") or it.get("query") or "").strip()
                    query = (it.get("query") or label).strip()
                    if query:
                        items.append({"label": label or query, "query": query})
        return items[:topk] if topk else items
    except Exception:
        return []


# 날짜 자동 보정 유틸 추가
def normalize_profile_dates(profile: dict) -> tuple[dict, dict]:
    """
    저장된 도착/출발 날짜가 과거이거나 역전되었을 때 안전하게 보정.
    - 도착일 < 오늘  → 오늘+1일로 보정
    - 출발일 ≤ 도착일 → 도착일+1일로 보정
    """
    today = date.today()
    arr = to_date(profile.get("arrival_date"), today + timedelta(days=1))
    # 출발일은 일단 arr+1 최소 보장으로 파싱
    dep = to_date(profile.get("depart_date"), arr + timedelta(days=1))

    fixed = {"fixed_past": False, "fixed_order": False}

    # 과거 보정
    if arr < today:
        arr = today + timedelta(days=1)
        fixed["fixed_past"] = True
    if dep < arr + timedelta(days=1):
        dep = arr + timedelta(days=1)
        fixed["fixed_order"] = True

    profile["arrival_date"] = arr.isoformat()
    profile["depart_date"] = dep.isoformat()
    return profile, fixed


# ---------- DB 유틸 (SQLite) ----------
def db_connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            q TEXT NOT NULL,
            a TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            city TEXT,
            days INTEGER,
            nights INTEGER,
            ppl INTEGER,
            with_kids INTEGER,
            weather_json TEXT,
            constraints_json TEXT,
            ts TEXT NOT NULL
        )
        """)
        conn.commit()

def insert_history(rec: Dict[str, Any]) -> None:
    with db_connect() as conn:
        conn.execute("""
        INSERT INTO history
        (q, a, profile_json, city, days, nights, ppl, with_kids, weather_json, constraints_json, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["q"],
            rec["a"],
            json.dumps(rec.get("profile", {}), ensure_ascii=False),
            rec.get("city"),
            int(rec.get("days", 0)) if rec.get("days") is not None else None,
            int(rec.get("nights", 0)) if rec.get("nights") is not None else None,
            int(rec.get("ppl", 0)) if rec.get("ppl") is not None else None,
            1 if rec.get("with_kids") else 0,
            json.dumps(rec.get("weather", {}), ensure_ascii=False),
            json.dumps(rec.get("constraints", {}), ensure_ascii=False),
            rec.get("ts") or datetime.now().isoformat(timespec="seconds"),
        ))
        conn.commit()

def fetch_history(q: str = "") -> List[Dict[str, Any]]:
    sql = "SELECT * FROM history"
    args: List[Any] = []
    if q:
        sql += " WHERE q LIKE ? OR a LIKE ? OR profile_json LIKE ?"
        like = f"%{q}%"
        args = [like, like, like]
    sql += " ORDER BY id DESC"
    with db_connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "q": r["q"],
            "a": r["a"],
            "profile": json.loads(r["profile_json"] or "{}"),
            "city": r["city"],
            "days": r["days"],
            "nights": r["nights"],
            "ppl": r["ppl"],
            "with_kids": bool(r["with_kids"]),
            "weather": json.loads(r["weather_json"] or "{}"),
            "constraints": json.loads(r["constraints_json"] or "{}"),
            "ts": r["ts"],
        })
    return out

def fetch_history_by_id(row_id: int) -> Optional[Dict[str, Any]]:
    with db_connect() as conn:
        r = conn.execute("SELECT * FROM history WHERE id=?", (row_id,)).fetchone()
    if not r: return None
    return {
        "id": r["id"],
        "q": r["q"],
        "a": r["a"],
        "profile": json.loads(r["profile_json"] or "{}"),
        "city": r["city"],
        "days": r["days"],
        "nights": r["nights"],
        "ppl": r["ppl"],
        "with_kids": bool(r["with_kids"]),
        "weather": json.loads(r["weather_json"] or "{}"),
        "constraints": json.loads(r["constraints_json"] or "{}"),
        "ts": r["ts"],
    }

def delete_history_row(row_id: int) -> None:
    with db_connect() as conn:
        conn.execute("DELETE FROM history WHERE id=?", (row_id,))
        conn.commit()

def delete_all_history() -> None:
    with db_connect() as conn:
        conn.execute("DELETE FROM history")
        conn.commit()

# ---------- 인덱스 확인 ----------
with st.status("인덱스 확인 중... (없으면 자동 생성)", expanded=False) as s:
    try:
        _ = RAGPipeline(k=2)
        s.update(label="인덱스 준비 완료 ✅", state="complete")
    except Exception as e:
        s.update(label=f"인덱스 준비 실패: {e}", state="error")

# DB 초기화
init_db()

# ---------- 프로필 ----------
profile = load_json(PROFILE_PATH, default_profile())

profile, _fix = normalize_profile_dates(profile)
if any(_fix.values()):
    st.info("저장된 여행 날짜가 지나가거나 역전되어 자동으로 보정했어요. 필요하면 사이드바에서 다시 변경해 주세요.")


# Langfuse user_id로 사용할 앱 세션 ID
if "app_session_id" not in st.session_state:
    st.session_state["app_session_id"] = str(uuid.uuid4())

# ---------- 사이드바: Tabs ----------
with st.sidebar:
    tab_new, tab_hist = st.tabs(["새 일정", "History"])

    # ── 새 일정 탭 ───────────────────────────────────────────────
    with tab_new:
        st.subheader("여행 일정 설정")

        # 날짜
        # default_arrival = to_date(profile.get("arrival_date"), date.today() + timedelta(days=1))
        # default_depart  = to_date(profile.get("depart_date"),  default_arrival + timedelta(days=2))

        # arrival_dt = st.date_input("시작일", value=default_arrival, min_value=date.today() + timedelta(days=1))
        # depart_dt  = st.date_input("종료일", value=max(default_depart, arrival_dt + timedelta(days=1)), min_value=arrival_dt + timedelta(days=1))

        default_arrival = to_date(profile.get("arrival_date"), date.today() + timedelta(days=1))
        default_depart  = to_date(profile.get("depart_date"),  default_arrival + timedelta(days=1))

        arrival_dt = st.date_input(
            "시작일",
            value=default_arrival,
            min_value=date.today() + timedelta(days=1),
        )
        depart_dt  = st.date_input(
            "종료일",
            value=max(default_depart, arrival_dt + timedelta(days=1)),
            min_value=arrival_dt + timedelta(days=1),
        )

        _calc, _err = compute_days(arrival_dt, depart_dt)
        if _err:
            st.error(_err)
        else:
            _days, _nights = _calc
            with st.container():
                st.markdown(f"**📅 여행 기간:** {arrival_dt:%Y-%m-%d} → {depart_dt:%Y-%m-%d}")
                st.markdown(f"**🧳 {_nights}박 {_days}일**")

        # 인원/아이동반/테마
        profile["ppl"]       = st.number_input("인원", 1, 10, int(profile.get("ppl", 2)))
        profile["with_kids"] = st.checkbox("아이 동반", value=bool(profile.get("with_kids", False)))

        theme_options = load_theme_options()
        default_focus_list = profile.get("focus_list", []) or (profile.get("focus","") or "").split("·")
        default_focus_list = [x for x in default_focus_list if x]
        selected_themes = st.multiselect(
            "핵심 테마/장소 (복수 선택 가능)",
            options=theme_options,
            default=[t for t in default_focus_list if t in theme_options],
        )
        profile["focus_list"] = selected_themes
        profile["focus"] = "·".join(selected_themes) if selected_themes else profile.get("focus","한옥·전통시장")

        # 프로필 저장/삭제
        c1, c2 = st.columns(2)
        with c1:
            if st.button("프로필 저장", use_container_width=True):
                profile["arrival_date"] = arrival_dt.isoformat()
                profile["depart_date"]  = depart_dt.isoformat()
                profile, _ = normalize_profile_dates(profile)  # 저장 전 최종 보정
                save_json(PROFILE_PATH, profile)
                st.success("프로필 저장 완료")

            # if st.button("프로필 저장", use_container_width=True):
            #     profile["arrival_date"] = arrival_dt.isoformat()
            #     profile["depart_date"]  = depart_dt.isoformat()
            #     save_json(PROFILE_PATH, profile)
            #     st.success("프로필 저장 완료")
        with c2:
            if st.button("프로필 삭제", use_container_width=True):
                try:
                    if os.path.exists(PROFILE_PATH):
                        os.remove(PROFILE_PATH)
                except Exception:
                    pass
                st.success("프로필을 삭제하고 기본값으로 초기화했습니다.")
                profile = default_profile()
                st.rerun()

        # RAG 재인덱싱
        if st.button("🔄 PDF여행자료 재추출(RAG)", use_container_width=True):
            with st.status("PDF → 청크 → 임베딩 → FAISS 저장", expanded=True) as s2:
                try:
                    vs, texts = build_faiss_from_folder(Path("data"), Path("vectorstore/faiss_index"))
                    # [ADD] 추천어 생성 → 메모리에 즉시 반영
                    from src.ingest import build_suggested_queries_from_pdfs
                    build_suggested_queries_from_pdfs(Path("data"), Path("data") / "suggested_queries.json")
                    st.session_state["suggest_keywords"] = load_suggest_keywords()
                    s2.update(label=f"재추출 완료 ✅ (청크 {len(texts)}개)", state="complete")
                except Exception as e:
                    s2.update(label=f"재추출 실패: {e}", state="error")

            # with st.status("PDF → 청크 → 임베딩 → FAISS 저장", expanded=True) as s2:
            #     try:
            #         vs, texts = build_faiss_from_folder(Path("data"), Path("vectorstore/faiss_index"))
            #         s2.update(label=f"재추출 완료 ✅ (청크 {len(texts)}개)", state="complete")
            #     except Exception as e:
            #         s2.update(label=f"재추출 실패: {e}", state="error")

        st.divider()

        # ✅ 사이드바 입력 + 버튼
        st.text_input(
            "원하는 여행을 입력하세요 (예: '전주 한옥마을 여행', '부산 여행' 또는 '9월은 어디로 가면 좋을까')",
            key="query_text",
        )
        #st.caption("지역이 없는 질문도 OK! (지역 미지정/광역 키워드면 하위 지역을 먼저 제안합니다)")

        if st.button("일정 생성", use_container_width=True, key="btn_generate"):
            st.session_state["selected_history_id"] = None  # 히스토리 상세 숨김
            st.session_state["trigger_generate"] = True
            st.rerun()

        # ── 추천 검색어 (자동/수동 모두 지원) ───────────────────────────
        st.subheader("추천 검색어 (RAG 기반)")

        def load_suggest_keywords(path: str = "data/suggested_queries.json", topk: int = 12):
            try:
                import json, os
                if not os.path.exists(path):
                    return []
                data = json.load(open(path, "r", encoding="utf-8"))
                items = []
                # 평면형/그룹형 모두 지원
                if isinstance(data, list):
                    for it in data:
                        label = (it.get("text") or it.get("label") or "").strip()
                        query = (it.get("query") or label).strip()
                        if label:
                            items.append({"label": label, "query": query})
                elif isinstance(data, dict) and isinstance(data.get("groups"), list):
                    for g in data["groups"]:
                        for it in g.get("items", []):
                            label = (it.get("label") or it.get("query") or "").strip()
                            query = (it.get("query") or label).strip()
                            if query:
                                items.append({"label": label or query, "query": query})
                return items[:topk] if topk else items
            except Exception:
                return []

        def _render_suggest_buttons(items: list[dict]):
            cols = st.columns(min(3, len(items)))
            for i, it in enumerate(items):
                label = it.get("label") or it.get("query") or ""
                query = it.get("query") or label
                with cols[i % len(cols)]:
                    if st.button(label, key=f"sg_{i}_{label}"):
                        st.session_state["pending_query_text"] = query
                        st.session_state["region_choice"] = ""
                        st.session_state["region_suggestions"] = []
                        st.session_state["selected_history_id"] = None
                        st.session_state["trigger_generate"] = True
                        st.rerun()

        # 1) 세션에 있으면 그대로 사용, 없으면 파일에서 로딩 시도
        if "suggest_keywords" not in st.session_state or not st.session_state["suggest_keywords"]:
            st.session_state["suggest_keywords"] = load_suggest_keywords()

        items = st.session_state.get("suggest_keywords", [])

        if items:
            _render_suggest_buttons(items)
        else:
            # ── 빈 상태 UI: 안내 + 즉시 생성(데이터 기반) + 기본(수동) 템플릿 ──
            st.info("추천어가 아직 없어요. 자료를 바탕으로 자동 생성하거나, 기본 템플릿을 사용해 시작할 수 있어요.")

            c1, c2 = st.columns([1, 1])
            with c1:
                # (권장) PDF/텍스트를 기반으로 자동 생성
                if st.button("📚 PDF 기반 추천어 만들기", use_container_width=True):
                    try:
                        from src.ingest import build_suggested_queries_from_pdfs
                        build_suggested_queries_from_pdfs(Path("data"), Path("data") / "suggested_queries.json")
                        st.session_state["suggest_keywords"] = load_suggest_keywords()
                        st.success("추천어를 생성했어요!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"추천어 자동 생성 실패: {e}")

            with c2:
                # (선택) 기존 “🔄 추천어 갱신” 유지하고 싶으면 남겨두기
                if st.button("🔄 추천어 갱신(LLM/RAG)", use_container_width=True):
                    try:
                        rag = RAGPipeline(k=5)
                        st.session_state["suggest_keywords"] = load_suggest_keywords()
                    except Exception as e:
                        st.warning(f"추천어 갱신 실패: {e}")
                        st.session_state["suggest_keywords"] = []


        # items = st.session_state.get("suggest_keywords", [])
        # if not items:
        #     st.caption("추천어가 아직 없습니다. PDF 자료를 추가 후 ‘PDF여행자료 재추출(RAG)’을 눌러 주세요.")
        # else:
        #     cols = st.columns(min(3, len(items)))
        #     for i, it in enumerate(items):
        #         label = it.get("label") or it.get("query") or ""
        #         query = it.get("query") or label
        #         with cols[i % len(cols)]:
        #             if st.button(label, key=f"sg_{i}_{label}"):
        #                 st.session_state["pending_query_text"] = query
        #                 st.session_state["region_choice"] = ""
        #                 st.session_state["region_suggestions"] = []
        #                 st.session_state["selected_history_id"] = None
        #                 st.session_state["trigger_generate"] = True
        #                 st.rerun()

        # if st.button("🔄 추천어 갱신", key="btn_refresh_suggest", use_container_width=True):
        #     try:
        #         # RAGPipeline 인스턴스 생성 후 프로필 기반 추천어 호출
        #         rag = RAGPipeline(k=5)
        #         # ↓ 실제 메서드 시그니처에 맞춰 필요시 인자(topk 등) 조정
        #         st.session_state["suggest_keywords"] = rag.suggest_queries(profile, topk=8)
        #     except Exception:
        #         st.session_state["suggest_keywords"] = []

        # keys = st.session_state.get("suggest_keywords", [])
        # if not keys:
        #     st.caption("아직 추천어가 없습니다. ‘추천어 갱신’을 눌러 보세요.")
        # else:
        #     cols = st.columns(min(3, len(keys)))
        #     for i, item in enumerate(keys):
        #         # item 이 문자열 또는 dict 인 두 경우 모두 지원
        #         if isinstance(item, dict):
        #             btn_text = item.get("label") or item.get("query") or ""
        #             query_to_use = item.get("query") or item.get("label") or ""
        #         else:
        #             btn_text = str(item)
        #             query_to_use = str(item)

        #         with cols[i % len(cols)]:
        #             if st.button(btn_text, key=f"sg_{i}_{btn_text}"):
        #                 st.session_state["pending_query_text"] = query_to_use
        #                 st.session_state["region_choice"] = ""
        #                 st.session_state["region_suggestions"] = []
        #                 st.session_state["selected_history_id"] = None
        #                 st.session_state["trigger_generate"] = True
        #                 st.rerun()


    # ── History 탭 (좌측 목록/버튼) ───────────────────────────────
    with tab_hist:
        st.subheader("History")
        q = st.text_input("이력 검색(입력 후 엔터)", "", key="history_search_sidebar")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("이력 새로고침", use_container_width=True):
                st.session_state["__history_refresh_token__"] = st.session_state.get("__history_refresh_token__", 0) + 1
        with c2:
            if st.button("전체 이력 삭제", type="secondary", use_container_width=True):
                with st.spinner("전체 이력 삭제 중..."):
                    delete_all_history()
                st.session_state["selected_history_id"] = None
                st.toast("전체 이력이 삭제되었습니다.", icon="🧹")

        _ = st.session_state.get("__history_refresh_token__", 0)
        with st.spinner("이력 조회 중..."):
            rows = fetch_history(q)
        st.caption(f"총 {len(rows)}건")

        for h in rows:
            with st.container(border=True):
                title = h.get("city") or h.get("q", "여행 일정")
                days  = h.get("days", 0) or 0
                nights = h.get("nights", max(0, (days or 1) - 1)) or 0
                ppl   = h.get("ppl", 1) or 1
                kids  = "있어요" if h.get("with_kids") else "아니오"
                ts    = h.get("ts", "") or ""

                prof = h.get("profile", {}) or {}
                arr  = (prof.get("arrival_date") or "").strip()
                dep  = (prof.get("depart_date")  or "").strip()

                st.markdown(f"**{title} {days}일 일정**")
                if arr and dep:
                    st.write(f"• 기간:{arr} ~ {dep}({nights}박{days}일)")
                else:
                    st.write(f"• 기간: - ~ -({nights}박{days}일)")
                st.write(f"• 인원:{ppl} / 아이동반:{kids}")
                st.caption(f"생성:{ts}")

                c3, c4 = st.columns(2)
                with c3:
                    if st.button("보기", key=f"hist_view_{h['id']}", use_container_width=True):
                        st.session_state["selected_history_id"] = h["id"]
                        st.rerun()
                with c4:
                    if st.button("삭제", key=f"hist_del_{h['id']}", type="secondary", use_container_width=True):
                        with st.spinner("삭제 중..."):
                            delete_history_row(h["id"])
                        if st.session_state.get("selected_history_id") == h["id"]:
                            st.session_state["selected_history_id"] = None
                        st.toast("삭제되었습니다.", icon="🗑️")
                        st.rerun()

# ---------- 메인 본문 ----------
st.subheader("🧭 AI 여행 플래너 본문")

# ✅ 프로젝트 소개(헤딩 레벨만 사용)
st.markdown("#### 프로젝트 소개")
st.markdown("###### 이 애플리케이션은 여행일정을 만들어주는 AI 여행플래너 입니다.")
st.markdown("###### 날씨, 여행코스, 맛집 체크를 진행하여 여행 일정을 생성해 줍니다.")
st.markdown("---")

# 상태 초기화
for key, default in [
    ("region_suggestions", []),
    ("region_choice", ""),
    ("last", ""),
    ("last_meta", {}),
    ("pending_query_text", ""),
    # ▼▼▼ 이 줄 추가 ▼▼▼
    ("suggest_keywords", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 앱 시작 시 파일에서 추천어 채우기(없으면 빈 리스트 유지)
if not st.session_state["suggest_keywords"]:
    st.session_state["suggest_keywords"] = load_suggest_keywords()

# === 생성 파이프라인 ===
# 카드에서 고른 추천값이 있으면 우선 사용(위젯 충돌 방지: query_text 직접 수정 금지)
query_val = (st.session_state.get("pending_query_text") or st.session_state.get("query_text") or "").strip()

if st.session_state.get("trigger_generate") and query_val:
    st.session_state["trigger_generate"] = False

    # 1) 지역 파싱
    region = parse_region_from_query(query_val)
    broad = is_broad_region(region)

    # 2) 지역 미지정 또는 광역 지역 → 추천(생성 보류)
    if not region or broad:
        month_hint = extract_month_from_text(query_val) or (
            to_date(profile.get("arrival_date"), date.today()).month
        )
        if not region:
            st.info("질문에 지역이 없어 국내 추천 지역을 먼저 제안합니다.")
            with st.spinner("추천 지역 분석 중..."):
                recs = llm_suggest_regions(
                    query=query_val,
                    month=month_hint,
                    with_kids=bool(profile.get("with_kids")),
                    themes=profile.get("focus_list", []),
                )
        else:
            st.info(f"'{region}'은 범위가 넓어요. 하위 지역을 먼저 선택해 주세요.")
            with st.spinner("하위 지역 추천 중..."):
                recs = llm_expand_broad_region(
                    broad=region,
                    month=month_hint,
                    with_kids=bool(profile.get("with_kids")),
                    themes=profile.get("focus_list", []),
                )
        st.session_state["region_suggestions"] = recs
        # 광역/미지정 상태에서는 아직 확정 도시를 프로필에 넣지 않음
    else:
        # 단일 도시라면 바로 프로필에 반영
        profile["city"] = region

    # 3) 확정 도시/기간(일단 계산)
    chosen = st.session_state.get("region_choice", "")
    city_final = (chosen or profile.get("city", "서울")).strip()

    _calc2, _ = compute_days(
        to_date(profile.get("arrival_date"), date.today() + timedelta(days=1)),
        to_date(profile.get("depart_date"),  date.today() + timedelta(days=3)),
    )
    if _calc2:
        profile["days"], profile["nights"] = _calc2
    else:
        profile["days"], profile["nights"] = 3, 2

    # 4) 추천 카드가 보이는 동안(=하위 지역 미선택) → 생성 보류
    if st.session_state["region_suggestions"] and not chosen:
        # 화면에서 추천 카드가 렌더되도록 하단 분기에서 처리
        pass
    else:
        # 5) 그래프 실행 + Langfuse 연결 (city_final 확정)
        if city_final:
            profile["city"] = city_final

        state = {
            "query": query_val,
            "city": profile["city"],
            "days": int(profile["days"]),
            "nights": int(profile["nights"]),
            "ppl": int(profile["ppl"]),
            "with_kids": bool(profile.get("with_kids")),
            "start_date": profile["arrival_date"],
            "end_date": profile["depart_date"],
            "themes": profile.get("focus_list", []),
            "retrieved": [],
            "course_json": {},
            "food_json": {},
            "final_md": "",
        }

        # Langfuse 준비
        lf_client, lf_cb, lf_trace = get_langfuse(
            session_name="travel_planner_langgraph",
            user_id=st.session_state["app_session_id"],
            metadata={
                "city": profile["city"],
                "ppl": profile["ppl"],
                "with_kids": profile["with_kids"],
                "start_date": profile["arrival_date"],
                "end_date": profile["depart_date"],
                "themes": profile.get("focus_list", []),
            },
        )
        invoke_config = (
            {"callbacks": [lf_cb], "run_name": "travel_planner_langgraph"}
            if lf_cb else {"run_name": "travel_planner_langgraph"}
        )

        with st.spinner("여행 일정을 생성하고 있어요..."):
            graph = build_graph()
            result = graph.invoke(state, config=invoke_config)

        # Langfuse trace에 요약 업데이트(옵션)
        try:
            if lf_trace:
                lf_trace.update(
                    output={
                        "final_md_present": bool(result.get("final_md")),
                        "retrieved_len": len(result.get("retrieved", [])),
                        "has_weather": bool(result.get("weather")),
                    }
                )
            if lf_client:
                lf_client.flush()
        except Exception:
            pass

        # 결과는 세션에만 저장(화면 직접 출력 금지)
        final_md = result.get("final_md", "")
        st.session_state["last"] = final_md
        st.session_state["last_meta"] = {
            "q": query_val,
            "profile": profile.copy(),
            "weather": result.get("weather", {}),
            "constraints": result.get("constraints", {}),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }

        # 새 생성 → 과거 선택 히스토리 숨김
        st.session_state["selected_history_id"] = None

        if not final_md:
            st.warning("생성된 일정이 비어 있습니다. 입력값(도시/날짜)을 확인해 주세요.")

        insert_history({
            "q": query_val, "a": final_md, "profile": profile,
            "city": profile["city"], "days": profile["days"], "nights": profile["nights"],
            "ppl": profile["ppl"], "with_kids": profile["with_kids"],
            "weather": result.get("weather", {}), "constraints": result.get("constraints", {}),
            "ts": datetime.now().isoformat(timespec="seconds")
        })

        # 선택/추천 상태 초기화
        st.session_state["region_suggestions"] = []
        st.session_state["region_choice"] = ""
        st.session_state["pending_query_text"] = ""  # 소비 완료

# (A) 추천 카드 렌더: 선택하면 자동 재생성 (사이드바 UI 없음)
if st.session_state.get("region_suggestions") and not st.session_state.get("region_choice"):
    st.markdown("### 추천 지역")
    cols = st.columns(2)
    for i, rec in enumerate(st.session_state["region_suggestions"], 1):
        with cols[(i-1) % 2]:
            st.markdown(f"**{rec['region']}**  \n<small>{rec.get('why','')}</small>", unsafe_allow_html=True)
            if st.button(f"이 지역으로 진행 ({rec['region']})", key=f"pick_{i}"):
                st.session_state["pending_query_text"] = f"{rec['region']} 여행"
                st.session_state["region_choice"] = rec["region"]
                st.session_state["region_suggestions"] = []
                st.session_state["trigger_generate"] = True
                st.rerun()

# ---- [단일 출력 경로] History '보기'가 있으면 그 내용만, 아니면 직전 결과 한 번만 출력 ----
if st.session_state.get("selected_history_id"):
    st.markdown("### 🗂️ 선택한 이력 보기")
    item = fetch_history_by_id(int(st.session_state["selected_history_id"]))
    if item:
        meta_line = f"{item.get('city','')} {item.get('days',0)}일 · 인원:{item.get('ppl',1)} · 아이동반:{'있음' if item.get('with_kids') else '아니오'} · {item.get('ts','')}"
        st.caption(meta_line)
        st.markdown(item.get("a") or "_(내용이 비어 있습니다)_")

        with st.expander("프로필 / 원천데이터 보기"):
            st.json(item.get("profile", {}))
            st.markdown("**날씨 요약**"); st.json(item.get("weather", {}))
            st.markdown("**제약(critic)**"); st.json(item.get("constraints", {}))
            st.markdown("**원본 질문**"); st.write(item.get("q",""))
else:
    if st.session_state.get("last"):
        st.markdown(st.session_state["last"])

        meta = st.session_state.get("last_meta", {})
        if meta:
            with st.expander("프로필 / 원천데이터 보기"):
                st.json(meta.get("profile", {}))
                st.markdown("**날씨 요약**"); st.json(meta.get("weather", {}))
                st.markdown("**제약(critic)**"); st.json(meta.get("constraints", {}))
                st.markdown("**원본 질문**"); st.write(meta.get("q", ""))

        st.divider()
