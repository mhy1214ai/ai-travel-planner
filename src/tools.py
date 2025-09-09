# src/tools.py
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

# --- 추가/보강: json 및 LangChain tool 가드 ---
import json

try:
    from langchain.tools import tool  # BaseTool 데코레이터
    HAS_LC_TOOL = True
except Exception:
    HAS_LC_TOOL = False
    # 데코레이터 폴백(툴 객체를 만들진 않지만, 심볼 정의용 no-op)
    def tool(_name=None):
        def _wrap(fn):
            return fn
        return _wrap
    
# httpx는 선택. 없으면 예보 API는 건너뛰고 시즌(평년값)으로 동작
try:
    import httpx  # type: ignore
    HAS_HTTPX = True
except Exception:
    HAS_HTTPX = False


# ─────────────────────────────────────────────────────────────
# (유지) 간단 날씨 툴 (기존 호환 인터페이스)
# ─────────────────────────────────────────────────────────────

def tool_weather(city: str, date: str) -> Dict[str, Any]:
    """
    1일 기준 간단 날씨 요약.
    - API 키/라이브러리가 있으면 get_weather_summary() 결과를 1일 범위로 사용
    - 아니면 기존 랜덤/월별 규칙으로 동작
    반환: {"city","date","summary","temp_c","wind_mps"}
    """
    s = _weather_one_day(city, date)
    if s:
        return s

    # (백업) 아주 단순한 월 기반 랜덤
    try:
        month = int(date.split("-")[1])
    except Exception:
        month = 6
    summary = "맑음" if month in (5, 6, 9, 10) else random.choice(["구름", "비", "맑음"])
    temp_c = 22 if summary == "맑음" else 19
    return {"city": city, "date": date, "summary": summary, "temp_c": temp_c, "wind_mps": 3}


# ─────────────────────────────────────────────────────────────
# (신규) 날씨 요약 / critic 가중치
#  - .env에 WEATHER_API_KEY가 있고 httpx가 설치되어 있으면 OpenWeather 예보 사용
#  - 아니면 월별 시즌(평년값)으로 동작
# ─────────────────────────────────────────────────────────────

PROVIDER = os.getenv("WEATHER_API_PROVIDER", "openweather").lower()
API_KEY = os.getenv("WEATHER_API_KEY", "")
TIMEOUT = float(os.getenv("WEATHER_API_TIMEOUT", "8"))
LANG = os.getenv("WEATHER_API_LANG", "kr")
UNITS = os.getenv("WEATHER_API_UNITS", "metric")  # metric|imperial|standard

def get_weather_summary(city: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    - 7일 이내: OpenWeather 5일/3시간 예보 → 범위 집계 → summary/temp_avg/rainy_days
    - 그 외/실패: 시즌(평년값) fallback
    """
    try:
        d0 = datetime.fromisoformat(start_date)
        days_ahead = (d0 - datetime.now()).days
    except Exception:
        days_ahead = 999

    if HAS_HTTPX and PROVIDER == "openweather" and API_KEY and days_ahead <= 7:
        geo = _owm_geocode(city)
        if geo:
            lat, lon = geo
            fc = _owm_forecast(lat, lon)
            if fc:
                agg = _summarize_forecast_for_range(fc, start_date, end_date)
                if agg:
                    season = _season_of(datetime.fromisoformat(start_date).month)
                    return {**agg, "season": season, "city": city, "start_date": start_date, "end_date": end_date}

    # fallback: 시즌(평년값)
    return _climate_baseline(city, start_date, end_date)


def make_weather_constraints(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    critic 가중치:
      - rainy_days ↑ → indoor_bias ↑, long_walk_penalty ↑
      - temp_avg ↑(여름) → long_walk_penalty 보정
    """
    indoor_bias = 0.2
    long_walk_penalty = 0.0
    if weather.get("rainy_days", 0) >= 2:
        indoor_bias = 0.6
        long_walk_penalty = 0.3
    if weather.get("temp_avg", 20) >= 28:
        long_walk_penalty = max(long_walk_penalty, 0.4)
    return {"indoor_bias": indoor_bias, "long_walk_penalty": long_walk_penalty}


# ─────────────────────────────────────────────────────────────
# 내부 구현(OpenWeather + 시즌 베이스라인)
# ─────────────────────────────────────────────────────────────

def _owm_geocode(city: str) -> Optional[Tuple[float, float]]:
    if not (HAS_HTTPX and API_KEY):
        return None
    base = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": API_KEY}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:  # type: ignore
            r = client.get(base, params=params)
            r.raise_for_status()
            data = r.json()
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None

def _owm_forecast(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    if not (HAS_HTTPX and API_KEY):
        return None
    base = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": UNITS, "lang": LANG}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:  # type: ignore
            r = client.get(base, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None

def _summarize_forecast_for_range(fc_json: Dict[str, Any], start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except Exception:
        return None

    slots: List[Dict[str, Any]] = fc_json.get("list", [])
    if not slots:
        return None

    sel = []
    for it in slots:
        dt_txt = it.get("dt_txt")
        if not dt_txt:
            continue
        try:
            it_dt = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if start <= it_dt < end:
            sel.append(it)
    if not sel:
        sel = slots

    temps: List[float] = []
    rainy_slots = 0
    cloudy_slots = 0
    clear_slots = 0

    for it in sel:
        main = it.get("main", {})
        weather = (it.get("weather") or [{}])[0]
        desc = (weather.get("main") or "").lower()
        cloud_pct = (it.get("clouds") or {}).get("all", 0)

        if "temp" in main:
            temps.append(float(main["temp"]))
        if "rain" in it or "drizzle" in desc or "rain" in desc:
            rainy_slots += 1
        elif cloud_pct >= 70 or "cloud" in desc:
            cloudy_slots += 1
        elif "clear" in desc:
            clear_slots += 1

    if not temps:
        return None

    temp_avg = round(sum(temps) / len(temps), 1)
    if rainy_slots >= max(2, len(sel) // 5):
        summary = "비"
    elif cloudy_slots >= clear_slots:
        summary = "구름많음"
    else:
        summary = "맑음"

    rainy_days = max(1, min((rainy_slots + 7) // 8, (end - start).days or 1))
    return {"summary": summary, "rainy_days": int(rainy_days), "temp_avg": temp_avg, "basis": "forecast"}

def _season_of(month: int) -> str:
    return ("winter", "spring", "summer", "autumn")[(month % 12) // 3]

def _climate_baseline(city: str, start_date: str, end_date: str) -> Dict[str, Any]:
    d0 = datetime.fromisoformat(start_date)
    season = _season_of(d0.month)
    temp_map = {"winter": 3, "spring": 17, "summer": 27, "autumn": 16}
    temp_avg = temp_map.get(season, 18)
    summary = "맑음" if season in ("spring", "autumn") else ("소나기" if season == "summer" else "구름많음")
    rainy_days = 2 if season == "summer" else 1
    return {
        "summary": summary,
        "rainy_days": rainy_days,
        "temp_avg": temp_avg,
        "season": season,
        "basis": "climate",
        "city": city,
        "start_date": start_date,
        "end_date": end_date,
    }


# ─────────────────────────────────────────────────────────────
# 보조: tool_weather가 API/시즌 요약을 1일 범위로 쓰도록 래핑
# ─────────────────────────────────────────────────────────────

def _weather_one_day(city: str, date: str) -> Optional[Dict[str, Any]]:
    try:
        start = datetime.fromisoformat(date)
        end = (start + timedelta(days=1)).date().isoformat()
        agg = get_weather_summary(city, date, end)
        if not agg:
            return None
        temp_c = round(float(agg.get("temp_avg", 20)))
        return {"city": city, "date": date, "summary": agg.get("summary"), "temp_c": temp_c, "wind_mps": 3}
    except Exception:
        return None

# 3) 지오코딩 함수 + Tool 래퍼
# [ADD] Kakao Local API
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "")
KAKAO_BASE = "https://dapi.kakao.com/v2/local/search/keyword.json"

def geocode_by_keyword(keyword: str) -> Dict[str, Any]:
    """
    Kakao Local API(있으면)로 키워드 지오코딩.
    환경변수 KAKAO_REST_API_KEY가 없거나 httpx 미설치 시 안전 폴백 반환.
    """
    if not KAKAO_REST_API_KEY or not HAS_HTTPX:
        return {
            "ok": False,
            "reason": "NO_HTTPX_OR_KAKAO_KEY",
            "items": [],
        }
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    try:
        with httpx.Client(timeout=TIMEOUT) as cli:
            r = cli.get(KAKAO_BASE, params={"query": keyword, "size": 5}, headers=headers)
            r.raise_for_status()
            data = r.json()
            items = []
            for doc in data.get("documents", []):
                items.append({
                    "name": doc.get("place_name"),
                    "addr": doc.get("road_address_name") or doc.get("address_name"),
                    "x": doc.get("x"),
                    "y": doc.get("y"),
                    "category": doc.get("category_group_name"),
                    "url": doc.get("place_url"),
                })
            return {"ok": True, "items": items}
    except Exception as e:
        return {"ok": False, "reason": str(e), "items": []}
    
    # # weather_summary_tool / geocode_tool 심볼은 항상 존재하도록 보장
    # if HAS_LC_TOOL:
    #     @tool("weather_summary_tool")
    #     def weather_summary_tool(city_or_query: str) -> str:
    #         try:
    #             res = get_weather_summary(city_or_query)
    #             return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
    #         except Exception as e:
    #             return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    #     @tool("geocode_tool")
    #     def geocode_tool(keyword: str) -> str:
    #         try:
    #             res = geocode_by_keyword(keyword)
    #             return json.dumps(res, ensure_ascii=False)
    #         except Exception as e:
    #             return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    # else:
    #     # LangChain Tool을 쓸 수 없을 때도 import 자체는 깨지지 않도록 None을 export
    #     weather_summary_tool = None
    #     geocode_tool = None

    # ─────────────────────────────────────────────────────────────
    # weather_summary_tool / geocode_tool 심볼은 항상 존재하도록 보장 (모듈 레벨)
    # ─────────────────────────────────────────────────────────────

    if HAS_LC_TOOL:
        # LangChain Tool 형태로 노출 (LangGraph ToolNode에서 바로 사용 가능)
        @tool("weather_summary_tool")
        def weather_summary_tool(city: str, arrival_date: str, depart_date: str) -> Dict[str, Any]:
            """
            여행 기간 날씨 요약 툴.
            - 입력: 도시명, 도착일(YYYY-MM-DD), 출발일(YYYY-MM-DD)
            - 출력(dict): {"summary","rainy_days","temp_avg","season","basis","city","start_date","end_date"}
            """
            return get_weather_summary(city, arrival_date, depart_date)

        @tool("geocode_tool")
        def geocode_tool(keyword: str) -> Dict[str, Any]:
            """
            키워드 지오코딩 툴(카카오 Local 사용 가능).
            - 입력: 키워드(예: "명동", "전주한옥마을")
            - 출력(dict): {"ok":bool,"reason":str,"items":[{name,addr,x,y,category,url}]}
            """
            return geocode_by_keyword(keyword)
    else:
        # LangChain Tool이 없어도 import 깨지지 않게 동일 이름의 callable 제공
        def weather_summary_tool(city: str, arrival_date: str, depart_date: str) -> Dict[str, Any]:
            return get_weather_summary(city, arrival_date, depart_date)

        def geocode_tool(keyword: str) -> Dict[str, Any]:
            return geocode_by_keyword(keyword)

    # 명시 export (선택이지만 권장)
    __all__ = [
        "tool_weather",
        "get_weather_summary",
        "make_weather_constraints",
        "geocode_by_keyword",
        "weather_summary_tool",
        "geocode_tool",
    ]

