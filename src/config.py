# src/config.py
from __future__ import annotations

import os
from typing import Optional, Tuple

# ------------ .env 로드 ------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# ============= Langfuse (v2/v3 호환) =============
try:
    from langfuse import Langfuse  # v2/v3 공통
    try:
        # v3
        from langfuse.callback import CallbackHandler as _LFCallback
    except Exception:
        # v2
        from langfuse import LangfuseCallbackHandler as _LFCallback  # type: ignore[attr-defined]
    LangfuseCallbackHandler = _LFCallback
except Exception:
    Langfuse = None
    LangfuseCallbackHandler = None


def get_langfuse(
    session_name: str = "travel_planner",
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Tuple[Optional["Langfuse"], Optional["LangfuseCallbackHandler"], Optional[object]]:
    """Langfuse client/callback/trace 일괄 준비 (.env에 키 없으면 None 반환)."""
    if Langfuse is None or LangfuseCallbackHandler is None:
        return None, None, None
    try:
        lf = Langfuse()  # LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST 자동 로드
        trace = lf.trace(name=session_name, user_id=user_id, metadata=metadata or {})
        try:
            cb = LangfuseCallbackHandler(trace_id=getattr(trace, "id", None))  # 일부 버전
        except TypeError:
            cb = LangfuseCallbackHandler()  # 일부 버전은 인자 없이 생성
        return lf, cb, trace
    except Exception:
        return None, None, None


# ============= Azure OpenAI (AOAI) =============
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-10-21")

DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O", "gpt-4o")
DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini")

# 임베딩 배포명: .env에서 여러 키를 지원 (우선순위대로)
EMBED_DEPLOY = (
    os.getenv("AOAI_EMBEDDING_DEPLOYMENT")        # 표준키 (있으면 이 값 사용)
    or os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")     # 당신의 .env 키
    or os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
    or os.getenv("AOAI_DEPLOY_EMBED_ADA")
    or "text-embedding-3-small"
)

# 개발 편의
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "").strip()

# ============= 청킹/인덱싱 파라미터 (.env에서 옵션) =============
def _to_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

CHUNK_SIZE = _to_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _to_int("CHUNK_OVERLAP", 200)
MIN_CHARS_PER_PAGE = _to_int("MIN_CHARS_PER_PAGE", 80)
MAX_PAGES_PER_DOC = _to_int("MAX_PAGES_PER_DOC", 120)
INGEST_WORKERS = _to_int("INGEST_WORKERS", 4)

# ============= 날씨 설정 (.env) =============
WEATHER_API_PROVIDER = os.getenv("WEATHER_API_PROVIDER", "openweather")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_TIMEOUT = _to_int("WEATHER_API_TIMEOUT", 8)
WEATHER_API_LANG = os.getenv("WEATHER_API_LANG", "kr")
WEATHER_API_UNITS = os.getenv("WEATHER_API_UNITS", "metric")

# ============= LangChain LLM/Embeddings =============
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    """개발용 랜덤 임베딩 (품질평가용으로는 권장X)."""
    def embed_documents(self, texts):
        import numpy as np
        return [np.random.default_rng(len(t)).random(1536).tolist() for t in texts]
    def embed_query(self, text):
        import numpy as np
        return np.random.default_rng(len(text)).random(1536).tolist()

def get_llm(model: str = "mini") -> AzureChatOpenAI:
    """
    Azure OpenAI Chat LLM
    - model="mini" -> DEPLOY_GPT4O_MINI
    - model="full" -> DEPLOY_GPT4O
    """
    return AzureChatOpenAI(
        openai_api_version=AOAI_API_VERSION,
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        deployment_name=DEPLOY_GPT4O_MINI if model == "mini" else DEPLOY_GPT4O,
        temperature=0.2,
    )

def get_embeddings():
    have_aoai = bool(AOAI_ENDPOINT and AOAI_API_KEY and EMBED_DEPLOY)
    if have_aoai:
        print(f"[emb] Using AzureOpenAIEmbeddings: deploy='{EMBED_DEPLOY}'")
        return AzureOpenAIEmbeddings(
            openai_api_version=AOAI_API_VERSION,
            azure_endpoint=AOAI_ENDPOINT,
            api_key=AOAI_API_KEY,
            model=EMBED_DEPLOY,
        )
    print("[emb] Using MockEmbeddings (fallback)")
    return MockEmbeddings()


