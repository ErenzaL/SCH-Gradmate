# SCH Graduation AI Chatbot 제출본

순천향대학교 AI·빅데이터학과 졸업요건에 대한 한국어 RAG 실습과 Streamlit 시연입니다. 검색은 FAISS·BM25·하이브리드 점수를 비교하며, 졸업학점·비교과 계산은 Python 규칙 엔진을 사용합니다. 수강과목 예시는 합성 데이터이며 학생 기록은 포함하지 않습니다.

## 환경 및 실행

Python 3.11 또는 3.12 가상환경을 만들고 의존성을 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sch-gradmate-venv --display-name 'SCH GradMate (.venv)'
```

프로젝트 루트에서 노트북을 실행합니다.

```bash
jupyter lab SCH_Graduation_RAG.ipynb
```

웹 시연을 실행합니다.

```bash
streamlit run app.py
```

Gemini를 사용하려면 `.env.example`을 참고해 `.env`를 만들거나 배포 서비스 Secrets에 `GEMINI_API_KEY`와 `GEMINI_MODEL`을 설정합니다. 키를 노트북 셀이나 저장소에 입력하지 마세요. API 한도·서비스 상태로 생성이 실패하면 근거 기반 대체 답변이 사용됩니다.

## 데이터 및 평가

`data/documents/`에는 제공된 대학 문서가 포함됩니다. `data/evaluation/questions.json`에는 30개 질문과 문서·쪽·청크 참조가 있으며, `data/evaluation/results/`에는 실제 검색 지표와 자동 답변 점검 결과가 있습니다. 검색 Recall은 수작업으로 지정한 문서·쪽 라벨에 대한 조건부 지표입니다. 자동 답변 점검은 의미적 사실 평가를 대체하지 않습니다.

현재 실험에서 하이브리드 검색은 Recall@5 0.944, MRR@5 0.872를 기록했습니다. 재정렬은 Recall@5를 바꾸지 않았고 MRR@5를 0.020 낮췄습니다. Gemini API는 평가 실행 당시 429 할당량 초과를 반환하여 실제 생성 응답의 사실성 평가는 미완료입니다. 이미지·스캔 페이지 OCR도 적용하지 않았습니다.

공식 배포는 하지 않았습니다. 상세 결과와 남은 한계는 `FINAL_REPORT_KO.md`를 확인하세요.
