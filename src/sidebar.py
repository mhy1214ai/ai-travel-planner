# sidebar.py
from typing import Dict, Any
import streamlit as st
from db import get_histories, delete_history_by_id, delete_all_histories

# 기존: render_input_form(), render_history_ui() 등은 그대로 두고
# 아래 History 탭 UI를 새로 작성
# - 좌측 사이드바에서 목록/검색/새로고침/전체삭제 처리
# - 항목 선택 시 st.session_state["selected_history_id"]에 저장

def _history_card(item: Dict[str, Any], idx: int):
    with st.container(border=True):
        title = f"{item.get('city','')} {item.get('days',0)}일 일정".strip()
        if not title or title.startswith(" "):
            title = "여행 일정"

        days = item.get("days", 0)
        nights = item.get("nights", max(0, days - 1))
        ppl = item.get("ppl", 1)
        kids = "있어요" if item.get("with_kids") else "아니오"
        ts = str(item.get("ts"))[:19] if item.get("ts") else ""

        st.markdown(f"**{title}**")
        st.caption(f"생성: {ts}")
        st.write(f"- 기간: {days}일 / {nights}박{days}일(표기)")
        st.write(f"- 인원: {ppl} / 아이동반: {kids}")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("보기", key=f"view_{idx}", use_container_width=True):
                st.session_state["selected_history_id"] = item["id"]
                # 우측 화면 업데이트 유도
                st.session_state["__need_rerun__"] = True
        with c2:
            if st.button("삭제", key=f"del_{idx}", type="secondary", use_container_width=True):
                delete_history_by_id(item["id"])
                st.session_state["__history_refresh_token__"] = st.session_state.get("__history_refresh_token__", 0) + 1
                st.toast("삭제되었습니다.", icon="🗑️")

def _render_history_sidebar():
    st.markdown("#### 이력 검색(입력 후 엔터)")
    query = st.text_input(
        "검색",
        key="history_query",
        placeholder="도시/질문으로 검색",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("이력 새로고침", key="btn_hist_refresh", use_container_width=True):
            st.session_state["__history_refresh_token__"] = st.session_state.get("__history_refresh_token__", 0) + 1
    with c2:
        # 두 단계 확인
        if st.button("전체 이력 삭제", key="btn_hist_delete_all", type="secondary", use_container_width=True):
            st.session_state["__confirm_delete_all__"] = True

    if st.session_state.get("__confirm_delete_all__"):
        with st.expander("정말 전체 이력을 삭제할까요?", expanded=True):
            c3, c4 = st.columns([1, 1])
            with c3:
                if st.button("예, 모두 삭제", key="btn_hist_delete_all_yes", use_container_width=True):
                    delete_all_histories()
                    st.session_state["__confirm_delete_all__"] = False
                    st.session_state["__history_refresh_token__"] = st.session_state.get("__history_refresh_token__", 0) + 1
                    st.toast("전체 이력이 삭제되었습니다.", icon="🧹")
            with c4:
                if st.button("아니오", key="btn_hist_delete_all_no", use_container_width=True):
                    st.session_state["__confirm_delete_all__"] = False

    # refresh 토큰이나 검색어가 바뀌면 목록 재조회
    _ = st.session_state.get("__history_refresh_token__", 0)
    items = get_histories(query)

    st.markdown(f"**총 {len(items)}건**")
    for idx, item in enumerate(items):
        _history_card(item, idx)

def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        tab1, tab2 = st.tabs(["새 일정", "History"])

        with tab1:
            # 기존 새 일정 폼
            from ui_new_trip_form import render_input_form  # 예: 기존 코드 경로
            render_input_form()

        with tab2:
            _render_history_sidebar()

    return {}
