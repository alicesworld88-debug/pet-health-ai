import os
import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("VERTEX_API_KEY")
_MODEL   = os.getenv("VERTEX_MODEL", "gemini-2.5-flash-lite")

_DEFAULT_SYSTEM_PROMPT = """당신은 반려견 건강 상담 AI입니다.
수의사 Q&A 데이터베이스에서 검색된 참고 답변을 바탕으로 보호자 질문에 답하세요.

규칙:
- 병명 확정, 처방 변경, 용량 조정은 절대 하지 않습니다
- 증상이 심각하면 반드시 수의사 상담을 권장합니다
- 따뜻하고 친절한 톤으로 답합니다
- 200자 이내로 간결하게 답합니다"""


def generate_answer(
    query: str,
    retrieved: list[dict],
    system_prompt: str | None = None,
) -> str:
    """
    query: 보호자 질문 문자열
    retrieved: Top-K 결과 리스트 — {'input': str, 'output': str, 'score': float}
    system_prompt: 에이전트별 시스템 프롬프트 (None이면 기본값 사용)
    반환: Gemini가 생성한 답변 문자열
    """
    url = (
        f"https://aiplatform.googleapis.com/v1/publishers/google/models"
        f"/{_MODEL}:generateContent?key={_API_KEY}"
    )

    ref_text = "\n\n".join(
        f"[참고 {i+1}]\nQ: {r['input']}\nA: {r['output']}"
        for i, r in enumerate(retrieved)
    )

    user_message = f"""보호자 질문: {query}

--- 참고 수의사 답변 ---
{ref_text}
-----------------------

위 참고 답변을 바탕으로 보호자 질문에 답해주세요."""

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt or _DEFAULT_SYSTEM_PROMPT}]
        },
        "contents": [{
            "role": "user",
            "parts": [{"text": user_message}]
        }]
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.Timeout:
        return "죄송합니다, 답변 생성에 시간이 걸리고 있어요. 잠시 후 다시 시도해 주세요."
    except requests.exceptions.HTTPError as e:
        return f"답변 생성 중 오류가 발생했습니다. ({e.response.status_code})"
    except (KeyError, IndexError):
        return "답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요."


# ── RAG 비교용: 검색 없이 LLM만으로 답변 ─────────────────────────────
_SOLO_SYSTEM_PROMPT = """당신은 반려견 건강 상담 AI입니다.
보호자 질문에 답하세요.
- 병명 확정, 처방 변경, 용량 조정은 하지 않습니다
- 증상이 심각하면 수의사 상담을 권장합니다
- 따뜻하고 친절한 톤으로 200자 이내로 답합니다"""


def generate_answer_solo(query: str) -> str:
    """
    검색(Retrieval) 없이 Gemini만으로 답변을 생성한다 (RAG 비교용 baseline).
    참고 수의사 Q&A를 전혀 주지 않으므로, RAG 대비 '검색의 효과'를 보여주는 대조군.
    """
    url = (
        f"https://aiplatform.googleapis.com/v1/publishers/google/models"
        f"/{_MODEL}:generateContent?key={_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": _SOLO_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": query}]}],
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.Timeout:
        return "죄송합니다, 답변 생성에 시간이 걸리고 있어요. 잠시 후 다시 시도해 주세요."
    except requests.exceptions.HTTPError as e:
        return f"답변 생성 중 오류가 발생했습니다. ({e.response.status_code})"
    except (KeyError, IndexError):
        return "답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요."
