"""
AI 수의사 채팅 파이프라인 — 멀티 에이전트 라우팅 구조

흐름:
    query → classify_intent → [SymptomAgent | TreatmentAgent | EmergencyAgent] → ChatResponse

확장 포인트:
    - classify_intent(): 규칙 기반 → ML 분류기로 교체 가능
    - 새 에이전트 추가: BaseAgent 상속 → agents 딕셔너리에 등록
    - Retriever Protocol: TF-IDF / BERT / 외부 검색엔진 교체 가능
    - generator.generate_answer(): 다른 LLM으로 교체 가능
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from utils.generator import generate_answer


# ── 데이터 구조 ────────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    """검색된 수의사 Q&A 1건."""
    input:     str
    output:    str
    score:     float
    lifecycle: str = ""
    disease:   str = ""


@dataclass
class ChatResponse:
    """채팅 파이프라인의 최종 출력."""
    query:   str
    intent:  str                        # 'symptom' | 'treatment' | 'emergency'
    answer:  str
    sources: list[RetrievedDoc] = field(default_factory=list)


# ── Retriever 인터페이스 ───────────────────────────────────────────────

@runtime_checkable
class Retriever(Protocol):
    """TF-IDF / BERT 등 모든 검색기가 구현해야 하는 인터페이스."""
    def match(self, query: str, top_k: int) -> tuple[list[int], list[float]]: ...


# ── 에이전트 시스템 프롬프트 ─────────────────────────────────────────────

_SYMPTOM_PROMPT = """당신은 반려견 증상 상담 AI입니다.
보호자의 증상 질문에 대해 수의사 Q&A를 참고하여 답변하세요.

규칙:
- 증상의 가능한 원인을 설명하되 병명을 확정하지 않습니다
- 언제 병원을 가야 하는지 조언합니다
- 처방이나 용량 조정은 절대 하지 않습니다
- 따뜻하고 친절한 톤으로 200자 이내로 답합니다"""

_TREATMENT_PROMPT = """당신은 반려견 처치·수술 정보 안내 AI입니다.
수의사 Q&A를 참고하여 처치·수술 관련 질문에 답변하세요.

규칙:
- 처치 방법, 회복 기간, 주의사항을 안내합니다
- 구체적 처방 변경이나 용량 조정은 하지 않습니다
- 담당 수의사와 상담을 권장합니다
- 200자 이내로 명확하게 답합니다"""

_EMERGENCY_PROMPT = """당신은 반려견 응급 상황 안내 AI입니다.
응급 상황에서 보호자가 즉시 해야 할 행동을 안내하세요.

규칙:
- 즉각적인 병원 방문이 최우선임을 반드시 강조합니다
- 이동 중 보호자가 할 수 있는 최소한의 안전 조치만 안내합니다
- 절대로 자가 치료를 권장하지 않습니다
- 100자 이내로 간결하고 빠르게 답합니다"""


# ── 에이전트 ────────────────────────────────────────────────────────────

class BaseAgent:
    """
    모든 에이전트의 기반 클래스.
    검색(retrieve) + 생성(generate) 흐름을 공유하고,
    system_prompt와 top_k만 하위 클래스에서 재정의합니다.
    """
    system_prompt: str = ""
    top_k:         int = 5

    def __init__(self, retriever: Retriever, corpus_df: pd.DataFrame):
        self.retriever = retriever
        self.corpus_df = corpus_df

    def _retrieve(self, query: str) -> list[RetrievedDoc]:
        indices, scores = self.retriever.match(query, top_k=self.top_k)
        return [
            RetrievedDoc(
                input=self.corpus_df.iloc[i]["input"],
                output=self.corpus_df.iloc[i]["output"],
                score=round(float(scores[j]), 4),
                lifecycle=self.corpus_df.iloc[i].get("lifeCycle", ""),
                disease=self.corpus_df.iloc[i].get("disease", ""),
            )
            for j, i in enumerate(indices)
        ]

    def run(self, query: str, intent: str) -> ChatResponse:
        docs   = self._retrieve(query)
        answer = generate_answer(
            query=query,
            retrieved=[{"input": d.input, "output": d.output, "score": d.score} for d in docs],
            system_prompt=self.system_prompt,
        )
        return ChatResponse(query=query, intent=intent, answer=answer, sources=docs)


class SymptomAgent(BaseAgent):
    """증상 질문 전담 에이전트."""
    system_prompt = _SYMPTOM_PROMPT
    top_k         = 5


class TreatmentAgent(BaseAgent):
    """처치·수술 질문 전담 에이전트."""
    system_prompt = _TREATMENT_PROMPT
    top_k         = 5


class EmergencyAgent(BaseAgent):
    """응급 상황 전담 에이전트 — top_k를 줄여 빠른 응답, 병원 방문 경고 강제 삽입."""
    system_prompt = _EMERGENCY_PROMPT
    top_k         = 3

    def run(self, query: str, intent: str) -> ChatResponse:
        resp = super().run(query, intent)
        resp.answer = "⚠️ 즉시 동물병원에 가세요!\n\n" + resp.answer
        return resp


# ── intent 분류 키워드 ──────────────────────────────────────────────────

_EMERGENCY_KEYWORDS = {'먹었어', '삼켰어', '응급', '쓰러', '경련', '발작', '피', '토혈'}
_TREATMENT_KEYWORDS = {'수술', '마취', 'mri', 'ct', '시술', '처치', '재활', '입원', '퇴원'}


# ── 파이프라인 (라우터) ────────────────────────────────────────────────

class ChatPipeline:
    """
    Intent 분류 → 전담 에이전트 라우팅.

        query
          ↓
        classify_intent()
          ↓
        agents[intent].run()
          ↓
        ChatResponse

    새 intent 추가 방법:
        1. BaseAgent를 상속한 새 에이전트 클래스 작성
        2. self.agents 딕셔너리에 등록
        3. classify_intent()에 분류 로직 추가
    """

    def __init__(self, retriever: Retriever, corpus_df: pd.DataFrame):
        self.agents: dict[str, BaseAgent] = {
            "symptom":   SymptomAgent(retriever, corpus_df),
            "treatment": TreatmentAgent(retriever, corpus_df),
            "emergency": EmergencyAgent(retriever, corpus_df),
        }

    def classify_intent(self, query: str) -> str:
        """
        규칙 기반 intent 분류.
        반환값: 'emergency' | 'treatment' | 'symptom'

        교체 방법: 이 메서드만 ML 분류기 호출로 바꾸면 됩니다.
        """
        q = query.lower()
        if any(k in q for k in _EMERGENCY_KEYWORDS):
            return "emergency"
        if any(k in q for k in _TREATMENT_KEYWORDS):
            return "treatment"
        return "symptom"

    def chat(self, query: str) -> ChatResponse:
        """질문 → intent 분류 → 전담 에이전트 → ChatResponse."""
        intent = self.classify_intent(query)
        return self.agents[intent].run(query, intent)


# ── 팩토리 함수 ────────────────────────────────────────────────────────

def build_pipeline(retriever_type: str = "bert") -> ChatPipeline:
    """
    파이프라인 한 줄 초기화.
    retriever_type: 'bert' (기본) 또는 'tfidf'
    """
    from utils.config import DATA_PROCESSED
    from utils.matcher import BERTMatcher, TFIDFMatcher

    corpus_df = pd.read_csv(DATA_PROCESSED / "corpus_preprocessed.csv")
    db_corpus = corpus_df["input_normalized"].fillna("").tolist()

    if retriever_type == "bert":
        retriever = BERTMatcher().load_or_build(db_corpus)
    elif retriever_type == "tfidf":
        tokens = corpus_df["input_tokens"].fillna("").tolist()
        retriever = TFIDFMatcher().fit(tokens)
    else:
        raise ValueError(f"retriever_type은 'bert' 또는 'tfidf'만 가능합니다: {retriever_type}")

    return ChatPipeline(retriever=retriever, corpus_df=corpus_df)


# ── 터미널 실행 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("파이프라인 초기화 중...")
    pipeline = build_pipeline(retriever_type="bert")
    print("준비 완료. 종료하려면 Ctrl+C\n")

    while True:
        try:
            query = input("질문: ").strip()
            if not query:
                continue
            resp = pipeline.chat(query)
            print(f"\n[intent: {resp.intent}]")
            print(f"답변: {resp.answer}\n")
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
