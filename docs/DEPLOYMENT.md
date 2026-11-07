# Render 배포 가이드 (Free 플랜)

> 상태: **배포 설정은 로컬에서 검증했지만 실제 Render 배포는 하지 않았습니다.** 아래 절차를 따라 배포한 뒤 10장의 점검표로 직접 확인하세요.

## 1. 구성 요약
| 항목 | 값 |
|---|---|
| 플랜 | Free: 512 MB RAM, 0.1 CPU, 15분 무트래픽 시 절전(재개에 약 1분), 디스크는 재시작마다 초기화 ([Render Free 문서](https://render.com/docs/free)) |
| Python | `3.11.15`(테스트한 버전). Render 기본 3.14 는 사용하지 않도록 `PYTHON_VERSION` + `.python-version` 으로 고정 ([문서](https://render.com/docs/python-version)) |
| 검색 | `RETRIEVAL_BACKEND=bm25` (최대 메모리 약 93 MB). 의미 검색은 약 0.9 GB 라 Free 에 올리지 않음 |
| 빌드 | `pip install -r requirements.txt && python scripts/build_index.py` |
| 시작 | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| 헬스체크 | `/_stcore/health` (Streamlit 내장, `ok` 와 HTTP 200 확인) |
| 보안 | CORS·XSRF 보호는 **켠 상태 유지**(`.streamlit/config.toml`), 업로드 5 MB, 오류 상세 숨김 |

## 2. 사전 준비
1. 저장소에 올릴 파일 확인: `render.yaml`, `.python-version`, `requirements.txt`, `.streamlit/config.toml`, `app.py`, `src/`, `scripts/`, `data/`(자료·규칙·전사본·`data/processed/rag_index/records.json`, `manifest.json`).
2. 올리면 안 되는 것: `.env`, `.venv/`, 실제 키. (`.gitignore` 에 포함, 임시 저장소로 동작 확인함)
3. `submission/` 폴더는 이번 변경이 반영되지 않은 옛 사본입니다. 올릴지 여부는 직접 결정하세요(배포에는 필요 없음).
4. 문서나 규칙을 바꿨다면 `python scripts/build_index.py` 를 다시 실행해 산출물을 갱신하고 커밋하세요. 테스트 `test_committed_artifact_matches_current_sources` 가 불일치를 알려 줍니다.

## 3. 명령어
```bash
# (로컬) 저장소 준비 — 아직 git 저장소가 아닙니다
git init && git add -A && git status        # .env 가 목록에 없는지 확인
git commit -m "SCH GradMate: Render deployment"
git remote add origin <본인 GitHub 저장소 URL>
git push -u origin main                      # 푸시는 직접 승인·실행하세요
```

## 4. Render 에서 만들기 (Blueprint)
1. Render 대시보드 → **New → Blueprint** → 저장소 선택 → `render.yaml` 이 감지됩니다.
2. 환경 변수 입력 화면:
   | 키 | 값 | 비고 |
   |---|---|---|
   | `PYTHON_VERSION` | `3.11.15` | 파일에 이미 있음 |
   | `RETRIEVAL_BACKEND` | `bm25` | 파일에 이미 있음 |
   | `GEMINI_API_KEY` | (선택) | 비워 두면 AI 요약 없이 자료 원문 발췌로 답합니다 |
   | `GEMINI_MODEL` | (선택) | AI Studio 에서 사용 가능한 모델 ID. 코드에 고정하지 않음 |
3. **Apply** → 빌드 로그에서 `chunks=99 dense=False` 가 보이면 색인 산출물 생성 성공입니다.
4. 배포 URL 로 접속해 질문합니다.

수동 생성 시: Web Service, Runtime Python, Build/Start Command 는 위 표와 같게, Health Check Path 는 `/_stcore/health`.

## 5. AI 요약(Gemini)에 대한 주의
- 이번 점검에서 설정된 모델(`gemini-3.8-flash`)이 **504 DEADLINE_EXCEEDED** 를 반환해 성공 응답을 확인하지 못했습니다. 키가 인증 단계는 통과했습니다. AI Studio 에서 모델 ID 와 할당량을 확인하세요.
- 호출이 실패해도 앱은 자료 원문 발췌로 답하며, 사용자에게 내부 오류 문구를 보이지 않습니다(테스트됨). 단 호출 제한 시간(15초) 동안 응답이 지연됩니다.
- 켜면 **질문과 검색된 자료 일부가 Google 로 전송**됩니다. 사이드바에 안내가 표시됩니다.

## 6. 무료 플랜 동작 특성
- 15분 동안 접속이 없으면 절전, 다음 접속 시 약 1분 지연. 첫 화면이 느릴 수 있습니다.
- 디스크가 초기화되므로 런타임 파일에 의존하지 않습니다. 업로드한 수강과목 파일은 메모리에서만 처리하고 저장하지 않습니다.
- 재시작 후에도 커밋된 `records.json`을 지문 검증 후 바로 사용합니다(검증: 재시작 후 헬스 200).

## 7. 문제 해결
| 증상 | 확인 |
|---|---|
| 빌드 실패 `Could not find a version` | `PYTHON_VERSION` 이 3.11.x 인지(3.14 로 빌드되면 휠이 없을 수 있음) |
| 메모리 초과(OOM) | `RETRIEVAL_BACKEND` 가 `bm25` 인지, `requirements-dense.txt` 를 쓰지 않았는지 |
| 시작은 되는데 화면이 계속 로딩 | 헬스체크 경로, 시작 명령의 `$PORT`·`0.0.0.0` |
| "자료를 불러오지 못했어요" | 빌드 로그의 `chunks=` 값, `data/documents/` 가 저장소에 있는지 |
| 웹소켓 연결 오류 | CORS/XSRF 를 끄지 마세요. 사용자 정의 도메인·프록시를 쓰면 `server.baseUrlPath`/도메인 설정을 확인 |

## 8. 로컬에서 Render 와 같은 방식으로 실행
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
RETRIEVAL_BACKEND=bm25 streamlit run app.py --server.port 8501 --server.address 0.0.0.0
curl -s localhost:8501/_stcore/health     # ok
```
개발 환경(노트북·테스트·의미 검색): `pip install -r requirements-dev.txt && pytest -q`.

## 9. 롤백
Render 대시보드 → Events/Deploys 에서 이전 배포를 선택해 Rollback 하거나, 문제가 된 커밋을 되돌려 푸시합니다.

## 10. 배포 후 직접 확인할 것 (아직 아무도 확인하지 않음)
- [ ] 공개 URL 접속, 첫 화면 로딩 시간
- [ ] "2024학번 졸업학점은?" → 130학점 + 근거 자료 펼치기
- [ ] "졸업학점은?" → 학번 되묻기, 이어서 "2024학번입니다"
- [ ] 휴대폰 화면에서 레이아웃·입력창·근거 펼침
- [ ] CSV 수강과목 업로드(양식 받기 → 올리기)
- [ ] 15분 방치 후 재접속(절전 복귀)
- [ ] 환경 변수에 키를 넣은 경우 AI 요약 성공 여부와 실패 시 대체 동작
