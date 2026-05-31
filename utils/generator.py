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

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
