# SCH GradMate — 순천향대학교 AI·빅데이터학과 졸업요건 RAG 챗봇

한국어 Streamlit 챗봇입니다. 제공된 학과 자료(PDF·이미지)에서 졸업학점, 다면적 졸업인증제, 봉사학습 등을 찾아 **자료명·쪽·적용 학번**과 함께 답합니다. 자료에 없는 내용은 만들지 않고 "확인하기 어렵습니다"라고 안내하며, 공식 졸업 승인 여부를 확정하지 않습니다.

> 점검 결과와 검증 현황은 [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md), 배포 절차는 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) 를 보세요. 점검 전 README 는 `docs/README.original_2026-11-02.md` 에 보존했습니다.

## 동작 방식

```
질문 ─ 입력 검증/주입 탐지 ─ 학번 추출(질문 > 선택값)
      ├─ 학점·점수 질문 → 구조화 규칙 엔진(data/graduation_rules.json): 결정론 계산, 학번 없으면 되묻기
      └─ 그 외 → 검색(BM25, 선택: +FAISS 의미 검색) → 근거 평가(supported / weak / none)
                  ├─ none  → "현재 제공된 자료만으로는 확인하기 어렵습니다…"
                  ├─ Gemini 설정됨 → 근거 [S1..]만 주고 요약, 실제 사용한 근거만 인용
                  └─ 그 외/실패 → 자료 원문 발췌(생성 문장 없음)
```

- 학번이 필요한 질문(졸업학점, 교양학점, 비교과 점수, TOPCIT…)에 학번이 없으면 먼저 묻습니다. "2024학번입니다"처럼 학번만 답해도 직전 질문에 이어 답합니다.
- 학번별 표에 없는 값을 다른 학번에 적용하지 않습니다(예: 2019학번에 TOPCIT 5점을 말하지 않음).
- 자료끼리 다르면 둘 다 출처와 함께 보여 줍니다(전공기초 28 vs 22학점, TOPCIT 회당 1점 vs 수준별 1~7점).
- 검색된 문서 안의 문장은 지시로 취급하지 않습니다. "이전 지시를 무시…" 같은 질문은 거절합니다.

## 실행

Python 3.11 권장(테스트한 버전).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # 런타임만 (Render 와 동일)
python scripts/build_index.py              # 색인 산출물 생성(자료가 바뀔 때)
streamlit run app.py
```

| 파일 | 용도 |
|---|---|
| `requirements.txt` | 배포(런타임) 의존성, 버전 고정 |
| `requirements-dense.txt` | 선택: 의미 검색(torch 등, 최대 약 0.9 GB) |
| `requirements-dev.txt` | 노트북·테스트·실험 전체 |
| `.env.example` | 환경 변수 예시(값은 비어 있음). 실제 `.env` 는 커밋하지 마세요 |

환경 변수: `RETRIEVAL_BACKEND`(`auto` \| `bm25`), `GEMINI_API_KEY`·`GEMINI_MODEL`(선택, 둘 다 있어야 AI 요약 사용), `RETRIEVAL_HEADERS=1`(선택, 색인 문맥 헤더).

## 자료와 한계

| 자료 | 처리 |
|---|---|
| PDF 4건 | PyMuPDF 텍스트·표 추출, 학번 구간별 행 보존, FAQ 문항 단위 청킹 |
| 이미지 3건 + 교육과정 PDF(스캔) | **OCR 아님.** 사람이 읽어 옮긴 전사본 `data/image_transcriptions.json` (검수 전 표시). 2024학번 표는 `graduation_rules.json` |
| 봉사 PDF 6쪽 | 예시 화면 캡처 — 미처리 |

- `졸업자격요건.png` 는 **학과명이 없습니다.** 복수전공 34학점·졸업논문·교양 32학점은 "AI·빅데이터학과 기준인지 확정할 수 없음" 주의문과 함께 안내합니다.
- 전공·교양 학점 표는 **2024학번만** 있습니다. 다른 학번은 학과 미표기 표로 부분 안내하거나 "확인 불가"로 답합니다.
- 청크마다 자료명, 쪽, 적용 학번, 학과, 문서 날짜(명시된 경우), 절 제목, 안정 chunk_id, 전사 여부, 중복 원본을 보존합니다.

## 수강과목·비교과 도구

- **수강과목 분석**: CSV/XLSX(필수 열 `course_name,credits,category,status`)를 올리면 이수 학점과 학번별 기준을 비교합니다. 파일은 메모리에서만 처리하고 5 MB 까지 받습니다. 검증된 과목 목록이 없어 전공필수 과목 충족은 자동 판정하지 않습니다.
- **비교과 활동 체크**: 해당 학번 표에 있는 활동만 보여 주고 **비교과지정 소계**(2019~2021: 11점, 2022: 21점, 2023~2026: 26점)와 비교합니다. 전체 합계(80/95점)에는 대학공통·교과·비교과미지정이 포함되어 학생 입력만으로 판정할 수 없습니다. 점수는 학생 입력에 따른 예상치이며 "충족"으로 확정하지 않습니다.

## 테스트와 평가

```bash
pip install -r requirements-dev.txt
pytest -q                                              # 142개, 외부 API 호출 없음(Gemini 는 mock)
python scripts/run_retrieval_experiments.py            # 검색 방식·청크 크기 비교 → data/evaluation/results/experiment_*.csv
python scripts/run_answer_evaluation.py                # 60문항 엔드투엔드 동작 평가(LLM 미사용)
python scripts/measure_resources.py                    # 메모리·지연 측정
```

- 평가 질문: `data/evaluation/questions.json`(60개: 원본 30 + 학번 되묻기·충돌·학번 혼용 함정·계산·미지원·주입). 원본은 `questions_v1_original.json`, 라벨 수정 이유는 각 항목의 `label_revision`.
- 요약(개발 집합, 낙관적): BM25 Recall@5 0.90 / MRR 0.88, 하이브리드와 통계적 차이 없음, 의미 검색만은 유의하게 낮음. 엔드투엔드 50/60 통과. 자세한 표와 해석은 감사 보고서 4~5장.
- 노트북 `notebooks/SCH_Graduation_RAG.ipynb`(54셀)는 원래 01~19장을 유지하고 20~26장을 추가했습니다. 원본은 `notebooks/SCH_Graduation_RAG.original_2026-11-02.ipynb`. 임베딩은 **학습·파인튜닝이 아니라** 사전학습 모델의 추론입니다.
- 사용 가능한 **생성형 답변 평가는 없습니다**(Gemini 성공 응답을 확인하지 못함).

## 배포

`render.yaml` 로 Render Free(512 MB)에 올릴 수 있게 구성했습니다(BM25 기본, 약 93 MB). **실제 배포·공개 URL 검증은 하지 않았습니다.** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) 참고.

## 알려진 한계
1. 이미지 전사는 사람 검수 전입니다.
2. 생성형(Gemini) 경로는 mock 테스트만 통과했고 실제 성공 호출은 확인하지 못했습니다(504).
3. 실제 브라우저·모바일 레이아웃, 파일 업로드 화면, Render 콜드 스타트 시간은 확인하지 못했습니다.
4. BM25는 같은 단어를 공유하는 FAQ 문항 사이에서 엉뚱한 발췌를 고를 수 있습니다(예: 봉사학습 신청 방법).
5. 평가 집합이 작고 개발 중 반복 사용되어 새 질문에서는 성능이 더 낮을 수 있습니다.
6. 공식 졸업 승인 도구가 아닙니다. 최종 확인은 학과 사무실에서 하세요.
