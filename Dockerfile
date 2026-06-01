# 공유 서빙 이미지 — TfidfFunction / BertFunction 둘 다 이 이미지를 사용.
#   - 분기는 env RETRIEVER_TYPE (tfidf|bert)
#   - TF-IDF 함수는 torch를 lazy import라 런타임에 로드하지 않음 (utils/matcher.py)
# 빌드: SAM이 자동 빌드. 수동: docker build -t pet-health-serving .
FROM public.ecr.aws/lambda/python:3.12

# 1) CPU 전용 torch 먼저 설치 (GPU 휠 회피 → 이미지 크기 축소)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) 나머지 의존성
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# 3) Sentence-BERT 모델을 이미지에 베이크 (콜드스타트에 HuggingFace 다운로드 회피)
ENV HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"
# 베이크 후에는 네트워크 접근 차단 (오프라인 로드)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# 4) 애플리케이션 코드
COPY app/    ${LAMBDA_TASK_ROOT}/app/
COPY utils/  ${LAMBDA_TASK_ROOT}/utils/
COPY chat.py ${LAMBDA_TASK_ROOT}/

# 임베딩(full_embeddings.npy)은 이미지에 넣지 않고 콜드스타트에 S3에서 /tmp로 로드
CMD ["app.lambda_handler.handler"]
