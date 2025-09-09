from __future__ import annotations

import json, re, random
from typing import TypedDict, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END

from .rag import RAGPipeline
from .config import get_llm, MOCK_MODE

# 날씨 유틸(있으면 사용)
try:
    from .tools import get_weather_summary, make_weather_constraints
except Exception:
    get_weather_summary = None
    make_weather_constraints = None

# [ADD] ReAct agent imports
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage

# 툴 임포트는 가드: 없으면 None
try:
    from .tools import weather_summary_tool, geocode_tool, HAS_LC_TOOL
except Exception:
    weather_summary_tool = None
    geocode_tool = None
    HAS_LC_TOOL = False

# ===== State =====
class PlannerState(TypedDict, total=False):
    # 입력
    query: str
    city: str
    days: int
    nights: int
    ppl: int
    with_kids: bool
    start_date: str
    end_date: str
    themes: List[str]

    # 중간 산출물
    retrieved: List[Dict[str, Any]]         # RAG 스니펫
    poi_seed: List[Dict[str, Any]]          # LLM으로 얻은 도시별 POI 시드
    weather: Dict[str, Any]
    constraints: Dict[str, Any]
    course_json: Dict[str, Any]             # {"days":[{"day":1,"title":"...","items":[...]}]}
    food_json: Dict[str, Any]               # {"days":[{"day":1,"places":[...]}]}
    errors: List[str]

    # 최종
    final_md: str


# ===== Helpers =====
def _append_error(state: PlannerState, msg: str) -> None:
    errs = state.get("errors") or []
    errs.append(msg)
    state["errors"] = errs


def _ensure_weather(state: PlannerState) -> None:
    """critic: 날씨 요약 + 제약 생성"""
    if state.get("weather"):
        return
    city = state.get("city") or ""
    sd = state.get("start_date") or ""
    ed = state.get("end_date") or ""
    wx = {}
    if get_weather_summary:
        try:
            wx = get_weather_summary(city, sd, ed)
        except Exception:
            wx = {}
    # ↓ 추가: 함수 임포트 불가 시, 툴을 직접 호출(LC Tool이 StructuredTool일 때는 .invoke 또는 직접 함수 호출 가능)
    elif weather_summary_tool and callable(weather_summary_tool):
        try:
            # StructuredTool일 경우: weather_summary_tool.run 또는 .invoke 사용
            # 여기서는 callable로 가정하고 직접 호출
            wx = weather_summary_tool(city, sd, ed)
        except Exception:
            wx = {}
    if not wx:
        # 간단 fallback (정보 부족시에도 파이프라인은 흐름 유지)
        wx = {
            "city": city, "start_date": sd, "end_date": ed,
            "summary": "맑음", "rainy_days": 0, "temp_avg": 20,
            "basis": "season", "season": None,
        }
    state["weather"] = wx

    cons = {}
    if make_weather_constraints:
        try:
            cons = make_weather_constraints(wx) or {}
        except Exception:
            cons = {}
    if not cons:
        rainy_days = int(wx.get("rainy_days", 0) or 0)
        temp_avg = float(wx.get("temp_avg", 20) or 20)
        cons = {
            "indoor_bias": 0.7 if rainy_days >= 2 else (0.4 if rainy_days == 1 else 0.2),
            "long_walk_penalty": 0.6 if temp_avg >= 29 else (0.2 if temp_avg <= 5 else 0.3),
        }
    state["constraints"] = cons


def _fmt_retrieved_snippets(docs: List[Dict[str, Any]], limit: int = 3) -> str:
    out = []
    for d in docs[:limit]:
        meta = d.get("meta", {}) or {}
        fn = meta.get("file_name") or meta.get("source") or "doc"
        pg = meta.get("page")
        txt = (d.get("content") or "").replace("\n", " ")
        out.append(f"- [{fn} p.{pg}] {txt[:220]}")
    return "\n".join(out)


def _safe_json_load(text: str) -> Dict[str, Any]:
    try:
        m = text.find("["); n = text.rfind("]")
        if m != -1 and n != -1 and n > m:
            return {"days": json.loads(text[m:n+1])}
        obj = json.loads(text)
        if isinstance(obj, list):
            return {"days": obj}
        return obj
    except Exception:
        return {}


# ---------- LLM 시드(POI) ----------
def llm_seed_pois(city: str, with_kids: bool, themes: List[str], indoor_bias: float) -> List[Dict[str, Any]]:
    """
    RAG가 빈약한 도시(예: 태안/남해 등)일 때 LLM으로 해당 도시의 대표 POI 12~16개를 먼저 수집.
    반환: [{name,type,indoor,area,why}]
    """
    try:
        llm = get_llm("mini")
    except Exception:
        return []

    theme_hint = ", ".join(themes) if themes else "없음"
    indoor_hint = "실내 위주" if indoor_bias >= 0.5 else "실내/야외 균형"
    prompt = f"""
당신은 지역 여행 데이터 큐레이터입니다.
도시 "{city}"의 대표 볼거리/체험/시장/박물관/자연 명소를 **12~16개** JSON 배열로 주세요.

규칙:
- 각 항목: {{"name":"...", "type":"박물관/시장/자연/사찰/카페/해변/전망/체험 등", "indoor":true/false, "area":"행정동/지명", "why":"핵심 이유 한 문장"}}
- 아이동반: {"예" if with_kids else "아니오"} → 아이동반일 때 **동선 편의/안전/실내**를 조금 더 고려
- 사용자 테마: {theme_hint} (도시와 맞지 않으면 비중 낮추고 대체)
- 현재 편향: {indoor_hint}
- 중복/모호 표현(“시립박물관”, “전통시장” 등 일반명사) 금지 → **실제 고유명** 사용
- 설명 없이 JSON 배열만 출력
"""
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        arr = json.loads(re.search(r"(\[.*\])", text, re.S).group(1) if re.search(r"(\[.*\])", text, re.S) else text)
        pois: List[Dict[str, Any]] = []
        for it in arr:
            name = str(it.get("name","")).strip()
            if not name: continue
            pois.append({
                "name": name,
                "type": str(it.get("type","")).strip() or "POI",
                "indoor": bool(it.get("indoor", False)),
                "area": str(it.get("area","")).strip(),
                "why": str(it.get("why","")).strip(),
            })
        return pois[:16]
    except Exception:
        return []


def _retrieve_context(state: PlannerState) -> None:
    """RAG retrieve + (필요 시) LLM POI 시드 확보"""
    if state.get("retrieved") and state.get("poi_seed"):
        return

    rag = RAGPipeline(k=5)
    q = (state.get("query") or "").strip()
    city = (state.get("city") or "").strip()
    cons = state.get("constraints", {}) or {}
    indoor = cons.get("indoor_bias", 0) >= 0.5

    aug = f"{city} {q}".strip()
    if indoor:
        aug += " 실내 박물관 미술관 온실 카페 쇼핑몰 전시 우천 대안"
    docs = rag.retrieve(aug)
    state["retrieved"] = docs

    # RAG가 약하면 LLM으로 POI를 먼저 수집
    if len(docs) < 2 and city:
        seed = llm_seed_pois(city, bool(state.get("with_kids")), state.get("themes", []), cons.get("indoor_bias", 0.0))
        state["poi_seed"] = seed
    else:
        state["poi_seed"] = state.get("poi_seed", [])


def _validate_course(days_req: int, data: Dict[str, Any]) -> Dict[str, Any]:
    """일정 JSON 검증/정제: 길이/중복/빈 아이템 제거"""
    out = {"days": []}
    seen = set()
    arr = data.get("days", [])
    if not isinstance(arr, list):
        return out

    for d in arr:
        day_no = int(d.get("day", 0) or 0)
        items = [i for i in (d.get("items") or []) if i.get("name")]
        # 이름 중복 제거
        uniq = []
        for i in items:
            key = (i.get("name"), i.get("area"))
            if key in seen: 
                continue
            seen.add(key)
            uniq.append(i)
        if not uniq:
            continue
        title = d.get("title") or f"{day_no}일차"
        out["days"].append({"day": day_no, "title": title, "items": uniq})

    # 일자 수 맞추기(부족하면 앞날 복제 or 초과시 잘라내기)
    out["days"].sort(key=lambda x: x.get("day", 0))
    if len(out["days"]) < days_req and out["days"]:
        src = out["days"][0]
        while len(out["days"]) < days_req:
            dup = {"day": len(out["days"])+1, "title": src["title"], "items": src["items"][:]}
            out["days"].append(dup)
    if len(out["days"]) > days_req:
        out["days"] = out["days"][:days_req]

    return out


def _compose_markdown(state: PlannerState) -> str:
    city = state.get("city", "-")
    days = int(state.get("days") or 0)
    nights = int(state.get("nights") or max(0, days - 1))
    ppl = int(state.get("ppl") or 1)
    kids = "예" if state.get("with_kids") else "아니오"

    wx = state.get("weather", {}) or {}
    basis = (wx.get("basis") or "season").lower()
    label = "예보" if basis == "forecast" else "시즌(평년값)"
    summary = wx.get("summary") or "-"

    sd = state.get("start_date") or "-"
    ed = state.get("end_date") or "-"

    course = state.get("course_json", {}) or {}
    food = state.get("food_json", {}) or {}
    errs = state.get("errors") or []

    lines = []
    lines.append(f"### {city} {days}일 일정")
    lines.append(f"시작일: {sd} ~ 종료일 : {ed} ({nights}박{days}일) / 인원: {ppl} / 아이동반: {kids} / 날씨: {summary} / 기준: {label}\n")

    if errs and not course.get("days"):
        lines.append("#### 생성 실패 안내")
        for e in errs:
            lines.append(f"- {e}")
        # POI 시드가 있으면 사용자에게 보여주기
        seeds = state.get("poi_seed") or []
        if seeds:
            lines.append("\n#### 참고 POI(자동 수집)")
            for s in seeds[:8]:
                lines.append(f"- {s.get('name')} ({s.get('type')}, {s.get('area')}) – {s.get('why')}")
        return "\n".join(lines)

    # 요약
    lines.append("#### 일정 요약")
    if course.get("days"):
        tops = []
        for d in course["days"]:
            title = d.get("title") or f"{d.get('day','?')}일차"
            first = (d.get("items") or [{}])[0].get("name", "")
            tops.append(f"{title} → {first}" if first else title)
        lines.append("- " + "\n- ".join(tops))
    else:
        lines.append("주요 일정 요약 없음")

    # 상세 일정 (중첩 리스트: 일차 → 항목)
    lines.append("\n#### 상세 일정")
    if course.get("days"):
        for d in course["days"]:
            day_no = d.get("day")
            title = d.get("title", f"{day_no}일차")
            items = d.get("items", [])

            # 상위: 일차(볼드) — 제목
            lines.append(f"- **{day_no}일차 — {title}**")

            if not items:
                # 하위: 들여쓰기 2공백 + 하이픈 → Markdown 중첩 리스트
                lines.append("  - 자유 일정 (추천 코스 없음)")
                continue

            for i in items:
                nm = i.get("name", "")
                why = i.get("why", "")
                if nm:
                    # 하위 불릿(들여쓰기 2공백)
                    if why:
                        lines.append(f"  - **{nm}** — {why}")
                    else:
                        lines.append(f"  - **{nm}**")
    else:
        lines.append("주요 일정 없음")


    # 맛집
    lines.append("\n#### 맛집")
    if food.get("days"):
        for d in food["days"]:
            day_no = d.get("day")
            lines.append(f"- **{day_no}일차**")
            for p in d.get("places", []):
                nm = p.get("name",""); ty = p.get("type",""); ar = p.get("area",""); why = p.get("why","")
                lines.append(f"  - {nm} ({ty}, {ar}) – {why}")
    else:
        lines.append("식사 계획 없음: 자유롭게 맛집을 탐방하세요!")

    # 팁
    lines.append("\n#### 팁")
    cons = state.get("constraints", {}) or {}
    if cons.get("indoor_bias", 0) >= 0.5:
        lines.append("- 우천/혹서 대비: **실내 동선** 중심으로 구성했어요.")
    if cons.get("long_walk_penalty", 0) >= 0.5:
        lines.append("- 도보 장거리 회피: 이동 거리를 짧게 유지했어요.")
    lines.append("- 출발 전 최신 **기상 예보**를 확인하세요.")

    return "\n".join(lines)

REACT_PLANNER_SYSTEM = """당신은 여행 플래너의 보조 계획 수립 에이전트입니다.
도구를 사용해 간단한 날씨/지오코딩 정보를 점검하고, 
최대 5줄의 요약 메모만 한국어로 작성하세요.

가용 도구(정확한 인자명을 지켜 호출):
- weather_summary_tool(city: str, arrival_date: str, depart_date: str) → {"summary","rainy_days","temp_avg","season","basis","city","start_date","end_date"}
- geocode_tool(keyword: str) → {"ok":bool,"items":[{name,addr,x,y,category,url}]}

규칙:
- 아이 동반, 우천 가능성 등 제약이 보이면 메모에 반영
- 출력은 반드시 마크다운 불릿 리스트 3~5개
- 툴 호출이 실패해도 가능한 정보로 합리적 제안을 하되, 추정임을 명시
"""



# ===== Nodes =====
def node_retrieve(state: PlannerState) -> PlannerState:
    _ensure_weather(state)
    _retrieve_context(state)   # fills retrieved + poi_seed
    return state

def node_critic(state: PlannerState) -> PlannerState:
    return state


def node_react_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm()

    # 사용할 수 있는 툴만 모으기
    # tools = [t for t in (weather_summary_tool, geocode_tool) if t]
    tools = [t for t in (weather_summary_tool, geocode_tool) if t is not None]

    # 인풋 메시지 구성
    city = state.get("city") or state.get("profile", {}).get("city")
    with_kids = state.get("with_kids")
    constraints = state.get("constraints", {}) or state.get("critic", {})
    base = state.get("query") or state.get("user_query") or state.get("question") or "여행 계획"

    sd = state.get("start_date") or ""
    ed = state.get("end_date") or ""

    human_input = (
        f"사용자 질의: {base}\n"
        f"도시(있으면): {city}\n"
        f"여행 기간: {sd} ~ {ed}\n"
        f"아이 동반 여부: {with_kids}\n"
        f"현재 제약(critic 요약): {constraints}\n"
        f"필요시 날씨/지오코딩 툴을 사용해 점검하고 3~5개의 계획 메모를 만들어줘.\n"
        f"(참고: weather_summary_tool 호출 시 city='{city}', arrival_date='{sd}', depart_date='{ed}' 형식 권장)"
    )

    try:
        if HAS_LC_TOOL and tools:
            prompt = ChatPromptTemplate.from_messages(
                [("system", REACT_PLANNER_SYSTEM),
                 MessagesPlaceholder("agent_scratchpad"),
                 ("human", "{input}")]
            )
            agent = create_react_agent(llm, tools, prompt)
            # ↓ 파싱 오류가 나도 런이 중단되지 않도록 옵션 추가
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
            out = executor.invoke({"input": human_input})

            memo = out.get("output", "").strip()
        else:
            # 폴백: 툴 없이도 LLM만으로 메모 생성(간단 버전)
            fallback_prompt = (
                REACT_PLANNER_SYSTEM +
                "\n(주의: 현재 도구 사용 불가. 합리적 추정임을 명시)\n\n" +
                human_input +
                "\n\n반드시 마크다운 불릿 3~5줄만."
            )
            resp = llm.invoke(fallback_prompt)
            memo = getattr(resp, "content", str(resp)).strip()
        return {"plan_notes": memo}
    except Exception as e:
        return {"plan_notes": f"- (참고) ReAct planner 실행 실패: {e}"}

    
def node_curator(state: PlannerState) -> PlannerState:
    """일자별 코스(JSON). RAG가 부족하면 poi_seed를 강하게 활용."""
    days = int(state.get("days") or 3)
    query = state.get("query") or ""
    city = state.get("city") or "한국"
    themes = ", ".join(state.get("themes") or [])
    cons = state.get("constraints", {}) or {}
    indoor = "예" if cons.get("indoor_bias", 0) >= 0.5 else "아니오"

    ctx = _fmt_retrieved_snippets(state.get("retrieved", []), limit=3)
    seeds = state.get("poi_seed") or []
    seed_lines = []
    for s in seeds[:12]:
        seed_lines.append(f"- {s.get('name')} ({s.get('type')}, {s.get('area')}) – indoor={str(bool(s.get('indoor'))).lower()} – {s.get('why')}")

    if not ctx and not seed_lines and not MOCK_MODE:
        # 근거가 전혀 없으면 즉시 실패 리턴
        _append_error(state, f"'{city}' 관련 자료가 부족해서 일정을 만들지 못했습니다. 검색어를 구체화하거나 하위 지역을 선택해 주세요.")
        state["course_json"] = {}
        return state

    prompt = f"""
당신은 여행 일정 큐레이터입니다. 아래 **근거**만 사용해 {days}일 코스를 JSON으로 생성하세요.

입력:
- 도시: {city}
- 사용자 의도: "{query}"
- 테마: {themes if themes else "없음"}
- 실내편향 필요: {indoor}

근거(RAG 요약):
{ctx if ctx else "- (RAG 없음)"}

근거(LLM 수집 POI):
{chr(10).join(seed_lines) if seed_lines else "- (POI 시드 없음)"}

요구사항:
- **반드시 days 배열 길이 = {days}**
- 각 day.items는 2~4개, **고유명**만 사용 (예: “시립박물관”, “전통시장” 같은 일반명사 금지)
- 동일 POI를 여러 날에 반복 금지
- 설명/말머리/코드블록 없이 **JSON만** 출력

JSON 스키마:
{{
  "days": [
    {{
      "day": 1,
      "title": "1일차 핵심 제목",
      "items": [
        {{"name":"천리포수목원","type":"수목원","indoor":false,"area":"소원면","why":"사계절 식물원, 비오면 온실 중심 관람"}},
        {{"name":"꽃지해변","type":"해변","indoor":false,"area":"안면도","why":"낙조 명소"}}
      ]
    }}
  ]
}}
"""
    try:
        if MOCK_MODE and not (ctx or seed_lines):
            # 개발모드: 아주 가벼운 목업
            days_list = []
            for d in range(1, days+1):
                days_list.append({"day": d, "title": f"{city} 코스 {d}일차", "items": []})
            data = {"days": days_list}
        else:
            llm = get_llm("mini")
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            data = _safe_json_load(text)
        data = _validate_course(days, data)
        if not data.get("days"):
            _append_error(state, "일정 JSON 생성에 실패했습니다.")
    except Exception:
        _append_error(state, "일정 생성 중 오류가 발생했습니다.")
        data = {}

    state["course_json"] = data
    return state


def node_food(state: PlannerState) -> PlannerState:
    """코스 주변 맛집 JSON. 근거가 전혀 없고 MOCK_MODE도 아니면 생성 중단."""
    course = state.get("course_json", {}) or {}
    days = course.get("days", [])
    city = state.get("city") or "한국"
    ppl = int(state.get("ppl") or 2)
    kids = "예" if state.get("with_kids") else "아니오"

    if not days:
        if MOCK_MODE:
            # 간단 목업
            out = {"days": []}
            for d in range(1, int(state.get("days") or 3)+1):
                out["days"].append({"day": d, "places": []})
            state["food_json"] = out
        else:
            state["food_json"] = {}
            _append_error(state, "맛집은 코스 생성 실패로 함께 생성하지 못했습니다.")
        return state

    poi_lines = []
    for d in days:
        names = ", ".join([i.get("name","") for i in d.get("items", [])])
        poi_lines.append(f"- {d.get('day','?')}일차: {names}")

    prompt = f"""
당신은 맛집/카페 큐레이터입니다.
도시 "{city}", 인원 {ppl}명, 아이동반: {kids}.

아래 일자별 코스 근처로 식당/카페 **2~3곳**씩 제안하고 JSON만 출력하세요.
- 고유명만 사용 (일반명사 금지)
- 중복 금지

코스 요약:
{chr(10).join(poi_lines)}

JSON:
{{
  "days": [
    {{
      "day": 1,
      "places": [
        {{"name":"현지 해산물 식당","type":"식당","area":"해변가","why":"지역 특산물"}},
        {{"name":"디저트 카페","type":"카페","area":"시내","why":"대표 디저트"}}
      ]
    }}
  ]
}}
"""
    try:
        llm = get_llm("mini")
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        data = _safe_json_load(text)
        # 간단 검증
        ok = bool(data.get("days"))
        if not ok:
            _append_error(state, "맛집 JSON 생성에 실패했습니다.")
            data = {}
    except Exception:
        _append_error(state, "맛집 생성 중 오류가 발생했습니다.")
        data = {}

    state["food_json"] = data
    return state


def node_manager(state: PlannerState) -> PlannerState:
    state["final_md"] = _compose_markdown(state)
    return state


# ===== Graph Builder =====
def build_graph():
    g = StateGraph(PlannerState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("critic", node_critic)
    # [ADD] 신규 노드
    g.add_node("react_planner", node_react_planner)
    g.add_node("curator", node_curator)
    g.add_node("food", node_food)
    g.add_node("manager", node_manager)
    # g.add_node("curator", node_curator)
    # g.add_node("food", node_food)
    # g.add_node("manager", node_manager)

    g.add_edge("retrieve", "critic")
 # [CHANGE] critic 다음에 react_planner를 거치도록 변경
    g.add_edge("critic", "react_planner")
    g.add_edge("react_planner", "curator")    
    # g.add_edge("critic", "curator")
    g.add_edge("curator", "food")
    g.add_edge("food", "manager")
    g.add_edge("manager", END)

    g.set_entry_point("retrieve")
    g.set_finish_point("manager")

    return g.compile()
