"""샘플 질문(손예시 + 네이버 실제 질문)에 대해 4방향 답변(/chat/compare)을 사전계산.

건강 상담 탭의 '4방향 답변 비교' 카드에 쓰일 결과를 미리 만든다.
run_api.py 서버가 떠 있는 상태에서 별도로 실행:

    PYTHONPATH=. python scripts/precompute_compare.py

→ data/processed/compare_results.json 생성 (app_builder가 자동 주입).

주의: 결과 JSON은 AI Hub 수의사 답변 원문을 포함 → 재배포 라이선스상
      깃허브에서 제외(.gitignore), S3/로컬에만 보관한다.
"""
import json
import time

import requests

from utils.app_builder import SAMPLE_SUGGESTIONS, build_naver

API = "http://localhost:8000/chat/compare"
OUT = "data/processed/compare_results.json"


def main():
    nav = build_naver()
    questions = list(SAMPLE_SUGGESTIONS) + [q for v in nav["samples"].values() for q in v]
    results = {}
    for i, q in enumerate(questions):
        for _ in range(4):                       # 네트워크 일시 오류 재시도
            try:
                results[q] = requests.post(API, json={"query": q}, timeout=120).json()
                break
            except Exception:
                time.sleep(3)
        print(f"{i + 1}/{len(questions)}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"DONE {len(results)}/{len(questions)} → {OUT}")


if __name__ == "__main__":
    main()
