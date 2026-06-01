"""
네이버 지식iN API로 반려견 Q&A 수집 — 챗봇 코퍼스 확장용
저장 포맷: input(질문), output(답변스니펫), intent, source
사용법: python collect_naver.py
"""
import os, time, re, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OUT_PATH      = Path("data/external/naver_questions.csv")

HEADERS = {
    "X-Naver-Client-Id":     CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
}

# 5차 추가 쿼리
QUERIES = [
    # 증상 - 세부
    ("강아지 눈꼽",              "symptom"),
    ("강아지 침 흘려",           "symptom"),
    ("강아지 코피",              "symptom"),
    ("강아지 혈뇨",              "symptom"),
    ("강아지 빈뇨",              "symptom"),
    ("강아지 변비",              "symptom"),
    ("강아지 헐떡거려",          "symptom"),
    ("강아지 황달",              "symptom"),
    ("강아지 고개 기울여",       "symptom"),
    ("강아지 뒷다리 마비",       "symptom"),
    ("강아지 디스크",            "symptom"),
    ("강아지 통증 신호",         "symptom"),
    ("강아지 두드러기",          "symptom"),
    ("강아지 알레르기",          "symptom"),
    ("강아지 비만",              "symptom"),
    ("강아지 식욕 없어요",       "symptom"),
    ("강아지 활력 없어요",       "symptom"),
    ("강아지 구취",              "symptom"),
    ("강아지 침 많이 흘려",      "symptom"),
    ("강아지 눈 안 떠",          "symptom"),
    ("강아지 눈 부음",           "symptom"),
    ("강아지 귀 부음",           "symptom"),
    ("강아지 발톱 이상",         "symptom"),
    ("강아지 발바닥 이상",       "symptom"),
    ("강아지 엉덩이 끌어",       "symptom"),
    ("강아지 항문 이상",         "symptom"),
    ("강아지 콧물",              "symptom"),
    ("강아지 재채기 피",         "symptom"),
    ("강아지 몸 냄새",           "symptom"),
    ("강아지 살 빠져",           "symptom"),
    # 처치 - 세부
    ("강아지 외이염 치료",       "treatment"),
    ("강아지 슬개골 수술 후",    "treatment"),
    ("강아지 수술 후 회복",      "treatment"),
    ("강아지 약 먹이는 방법",    "treatment"),
    ("강아지 주사 맞히기",       "treatment"),
    ("강아지 수액 치료",         "treatment"),
    ("강아지 보험 추천",         "treatment"),
    ("강아지 심장병 치료",       "treatment"),
    ("강아지 신부전 치료",       "treatment"),
    ("강아지 암 치료",           "treatment"),
    ("강아지 당뇨 관리",         "treatment"),
    ("강아지 관절염 치료",       "treatment"),
    ("강아지 호르몬 치료",       "treatment"),
    ("강아지 안약 넣기",         "treatment"),
    ("강아지 귀 세척",           "treatment"),
    # 응급 - 세부
    ("강아지 우유 먹었어요",     "emergency"),
    ("강아지 커피 먹었어요",     "emergency"),
    ("강아지 아보카도",          "emergency"),
    ("강아지 자일리톨 먹었어요", "emergency"),
    ("강아지 날고기 먹었어요",   "emergency"),
    ("강아지 비닐 먹었어요",     "emergency"),
    ("강아지 발열 응급",         "emergency"),
    ("강아지 화상",              "emergency"),
    ("강아지 뱀 물렸어요",       "emergency"),
    ("강아지 벌 쏘였어요",       "emergency"),
    ("강아지 파보 증상",         "emergency"),
    ("강아지 디스템퍼",          "emergency"),
    ("강아지 복수",              "emergency"),
    ("강아지 기흉",              "emergency"),
    ("강아지 장염 응급",         "emergency"),
]
    # 품종별 증상
    ("말티즈 증상",          "symptom"),
    ("푸들 증상",            "symptom"),
    ("치와와 증상",          "symptom"),
    ("포메라니안 증상",      "symptom"),
    ("비숑 증상",            "symptom"),
    ("골든리트리버 증상",    "symptom"),
    ("래브라도 증상",        "symptom"),
    ("시바견 증상",          "symptom"),
    ("코기 증상",            "symptom"),
    ("닥스훈트 증상",        "symptom"),
    # 나이별 증상
    ("강아지 노령견 증상",   "symptom"),
    ("강아지 새끼 증상",     "symptom"),
    ("강아지 노견 치료",     "treatment"),
    ("강아지 어린 수술",     "treatment"),
    # 구체적 질병
    ("강아지 파보바이러스",  "emergency"),
    ("강아지 홍역",          "symptom"),
    ("강아지 켄넬코프",      "symptom"),
    ("강아지 렙토스피라",    "emergency"),
    ("강아지 광견병",        "emergency"),
    ("강아지 심장사상충 증상", "symptom"),
    ("강아지 슬개골 탈구",   "symptom"),
    ("강아지 고관절 이상",   "symptom"),
    ("강아지 척추 이상",     "symptom"),
    ("강아지 안구돌출",      "emergency"),
    ("강아지 백내장",        "symptom"),
    ("강아지 녹내장",        "symptom"),
    ("강아지 외이염",        "symptom"),
    ("강아지 중이염",        "symptom"),
    ("강아지 치주염",        "symptom"),
    ("강아지 췌장염",        "symptom"),
    ("강아지 장중첩",        "emergency"),
    ("강아지 위확장",        "emergency"),
    ("강아지 방광결석",      "symptom"),
    ("강아지 신부전",        "symptom"),
    ("강아지 간부전",        "symptom"),
    ("강아지 쿠싱증후군",    "symptom"),
    ("강아지 당뇨병",        "symptom"),
    ("강아지 갑상선",        "symptom"),
    ("강아지 빈혈",          "symptom"),
    ("강아지 림프종",        "symptom"),
    # 다양한 표현
    ("멍멍이 아파요",        "symptom"),
    ("우리 개 이상해요",     "symptom"),
    ("반려견 병원",          "treatment"),
    ("반려견 응급",          "emergency"),
    ("반려견 수술",          "treatment"),
    ("반려견 치료비",        "treatment"),
    ("반려동물 증상",        "symptom"),
    ("개 먹었어요",          "emergency"),
    ("개 삼켰어요",          "emergency"),
    ("개 쓰러졌어요",        "emergency"),
]

DISPLAY   = 100
MAX_START = 901
DELAY     = 0.1


def clean(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def fetch(query: str, start: int) -> list[dict]:
    r = requests.get(
        "https://openapi.naver.com/v1/search/kin.json",
        headers=HEADERS,
        params={"query": query, "display": DISPLAY, "start": start, "sort": "sim"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("items", [])


# 기존 데이터 로드해서 중복 방지
rows = []
seen = set()
if OUT_PATH.exists():
    existing = pd.read_csv(OUT_PATH)
    seen = set(existing["query"].tolist())
    rows = existing.to_dict("records")
    print(f"기존 데이터 로드: {len(rows):,}건 (중복 방지)")

total_calls = 0

for query, intent in QUERIES:
    print(f"\n[{intent}] {query}")
    count = 0
    for start in range(1, MAX_START + 1, DISPLAY):
        try:
            items = fetch(query, start)
            total_calls += 1
            new = 0
            for item in items:
                q = clean(item.get("title", ""))
                a = clean(item.get("description", ""))
                # 반려견 관련 질문만 필터 (강아지/고양이/반려 포함)
                if not q or q in seen or len(q) < 8:
                    continue
                if not any(k in q+a for k in ["강아지","반려견","반려동물","멍","댕","개가","개를","개의"]):
                    continue
                seen.add(q)
                # title + description 합쳐서 완전한 질문 텍스트 구성
                full_q = (q + " " + a).strip() if a else q
                rows.append({
                    "query":  full_q,
                    "intent": intent,
                    "source": "naver_kin",
                })
                new += 1
            count += new
            print(f"  start={start:4d} | 수신 {len(items):3d}건 | 신규 {new:3d}건 | 누적 {len(rows):,}건")
            if len(items) < DISPLAY:
                break
            time.sleep(DELAY)
        except Exception as e:
            print(f"  ✗ {e}")
            break
    print(f"  → {query}: {count}건 추가")

df = pd.DataFrame(rows)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n{'='*55}")
print(f"수집 완료: {len(df):,}건 | API 호출 {total_calls}회/25,000")
print(df["intent"].value_counts().to_string())
print(f"저장: {OUT_PATH}")
