# temporal-vault-rag

날짜가 붙은 Markdown 볼트에서 시점을 고려해 검색하는 RAG 실습 레포. 계획·진행 상태의 단일 상태판은 상위 폴더의 `PLAN.md`이며, 이 파일은 레포 안에서 일할 때 필요한 것만 적는다.

## 명령 (Windows, 레포 루트 기준)
- 환경 활성화: `.venv\Scripts\Activate.ps1` (Python 3.12, CPU 전용 torch)
- 재설치: `py -3.12 -m venv .venv` → `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` → `pip install -r requirements.txt` → `pip install -e . --no-deps`
- `.env` 만들기: `.env.example` 복사. 모델 캐시는 `HF_HOME`(레포 밖 `../models/hf`)만 쓴다. `SENTENCE_TRANSFORMERS_HOME`을 넣으면 sentence-transformers가 다른 캐시를 찾아 모델 로딩이 실패한다.
- 모델 받기: `python scripts/download_model.py` (bge-m3 dense 가중치만, ONNX·colbert·sparse 제외, 약 2.2GB)
- 테스트: `python -m pytest -q` / 모델 로딩 제외: `python -m pytest -q -m "not slow"` / 오프라인: `$env:HF_HUB_OFFLINE=1`
- 콘솔 한글이 깨지면 `$env:PYTHONUTF8=1` (출력 인코딩 문제일 뿐 동작에는 영향 없음)
- 데모 전체 재현(청킹 → 인덱스 → 사실표 → 평가): `python -m tvrag.run --config configs/demo.yaml --modes A B C CL D DL`. 결과는 `demo/results/<날짜>_<모드>.{json,md}`와 비교표 `<날짜>_<모드들>.md`, 빌드물은 `demo/build/`(gitignore). CPU에서 2-3분. `--modes` 기본값은 A B.
- 사실표만: `python -m tvrag.facts --config ... [--candidates <json>]`(관측 파일 → 표 `.jsonl`·`.md`, 규칙 후보·적중률). 데모 관측 파일은 `scripts/write_demo_facts.py`로 다시 쓴다(청크가 있어야 함).
- 실제 볼트: `python -m tvrag.run --config configs/local.yaml` (청크가 있으면 재사용, 인덱스는 manifest가 청크 해시·모델·토크나이저 설정과 다를 때만 다시 만듦. 644청크 임베딩 CPU 약 9분). 평가셋은 `paths.eval_set` 또는 `--eval <파일>`. `--no-eval`은 인덱스까지만. `--date <태그>`가 결과 파일 이름 앞부분.
- 단계별: `python -m tvrag.chunk|index|facts|eval --config ...`, 검색 확인: `python -m tvrag.retrieve --config ... -m C -k 5 --ref 2025-06-30 "질문"`(`-m D`는 사실표 답을 먼저 출력)
- 보조 스크립트: `scripts/tune_c.py`(모드 C 설정 비교, 데모 평가셋 전용), `scripts/report_date_ranges.py`(created 청크의 날짜 구간 보고), `scripts/failure_cases.py`(결과 JSON → 실패 사례·원인 분류). 뒤 둘은 본문을 인용하므로 실제 볼트에는 비공개 폴더로만 출력한다.

## 모듈과 데이터 흐름 (`src/tvrag/`)
`config` → `ingest` → `chunk` → `index` → `retrieve` → `eval`, 모드 D는 `facts`가 추가로 붙는다.
- `config`: `.env`를 읽어 HF 캐시 경로를 먼저 지정하고 YAML 설정을 읽는다. HF를 쓰는 모듈보다 먼저 import한다. 상대 경로는 레포 루트 기준.
- `ingest`: 스냅샷 폴더의 Markdown → 문서(프론트매터 + 본문).
- `chunk`: 헤딩 단위 청크. 사건 시각은 날짜 헤딩 → 프론트매터 `date` → `created` 순(`event_date`, `date_source`). 여기에 `dates`가 본문 날짜 언급으로 `date_min`·`date_max`(내용 시점 구간)를 붙인다.
- `dates`: 날짜 언급 추출. 형식은 `yyyy-mm-dd`, `m/d`, `m월 d일`(스냅샷 조사 결과). 링크 안 날짜, 숫자 나열·소수·횟수 단위 옆 `m/d`는 제외. 연도 없는 날짜는 기준일(event_date와 본문 완전한 날짜 중 가장 늦은 날)에 가장 가까운 해. 구간 = 언급 + 실제 기록일(헤딩·프론트매터 date, created는 제외), 말뭉치 마지막 기록일 이후 언급(목표·계획)은 제외, 언급이 없으면 event_date. 규칙과 근거는 모듈 docstring.
- `index`: bge-m3 임베딩을 LanceDB에, kiwipiepy 형태소로 BM25 인덱스. LanceDB를 고른 이유: 파일 기반 임베디드 저장소라 서버가 없고, 메타데이터(사건 시각) 컬럼에 where 필터를 걸 수 있어 as-of 검색에 맞다. 임베딩·BM25 입력은 `문서제목 > 헤딩경로` + 본문(`passage_text`). BM25는 내용 형태소(NNG·NNP·NR·VV·VA·XR·SL·SH·SN)만 남긴다: 조사·어미·접미사·의존명사는 어디에나 있어 잡음이고, 단위·숫자는 질문이 묻는 값이라 남긴다. 인덱스 폴더: `lancedb/`, `bm25_tokens.jsonl`, `manifest.json`(타임스탬프 없음, 같은 입력이면 같은 내용. `vectors_sha256`로 재현성 확인).
- `retrieve`: 공통 인터페이스 `retrieve(question, mode, k, as_of=None, *, qtype=None, reference_date=None) -> [Hit]`(Hit에 모드·점수·근거 청크·날짜 구간). 모드 A 벡터(LanceDB 코사인, 전수 검색), B는 벡터와 BM25 각 상위 `retrieve.candidates`개를 RRF(`rrf_k`=60)로 합침. C는 B 위에 질문 유형별로: 일반 = B, 시점 고정 = `date_min > as_of` 사전 필터, 현재 값 = B 후보를 `0.5 ** (경과일 / 60)`(경과일 = 기준일 - date_max)로 재정렬, 변화 추적 = B 후보 30개에서 date_max가 다른 청크를 관련도 순으로 고른 뒤 시간순 정렬. 설정은 `retrieve.c`(데모로 정해 고정, 근거는 `scripts/tune_c.py`). `qtype`을 주면 질문 해석 대신 그 유형을 쓴다(평가의 "정답 라벨 C"). D = C + 사실표(S6): 같은 질문 해석으로 사실표 행을 골라 그 근거 청크를 순위 앞에 두고 C 목록으로 채운다. 사실 답(값·유효 구간·조건·성격·근거)은 1위 Hit의 `components["facts"]`. 사실표가 없거나 속성을 못 찾거나 일반 질문이면 D = C.
- `query`: 모드 C의 질문 해석 규칙. 유형 우선순위는 변화 단서 > 날짜 표현(시점 고정) > 지금·요즘 단서(현재 값) > 일반. 기간은 끝날로(초 10일, 중순 20일, 말 말일), 연도 없으면 기준일 연도(기준일보다 뒤면 전년), 상대 표현(N주 전, 지난달)은 기준일에서 뺀다. 개발용 질문은 `tests/test_dates_query.py`.
- `facts`: 이중 시간 사실표 `(대상, 속성, 값, 단위, 측정조건·계열, 성격, 유효시작, 유효끝, 근거청크, 기록시각)`. 사람이 정리한 관측 파일(JSONL)에서 만든다(규칙 후보 `extract_candidates`가 보조, LLM 없음). 기록 순서로 재생하며 새 값은 같은 (대상, 속성, 계열) 체인의 이전 행 유효끝만 채우고 지우지 않는다. 제안·추정 행은 별도 체인이라 측정·결정 값을 닫지 않고 답에는 '참고'로만 나온다. 기록시각 = 근거 청크 날짜(유효시작보다 이르면 유효시작). 속성 사전(`ATTR_KEYWORDS`)은 스키마와 볼트 어휘로 만들었다(평가 질문 미참조). 유형별 행 선택: 시점 고정 = as_of에 유효한 행(기본 계열 + `facts.change_series` 중 가장 늦게 시작한 것), 현재 값 = 기준일에 유효한 기본 계열(`facts.default_series`) 행, 변화 추적 = 기본 + change_series 전체를 시간순. 근거 순서: 시점 고정은 기록일 세션 청크 먼저(as_of 뒤에 쓴 청크 제외), 현재 값은 그 값을 가장 늦게 적은 청크 먼저, 변화 추적은 가장 많은 행을 덮는 청크부터 고르는 탐욕 커버. 설정은 `facts:` 절, 경로는 `paths.facts_source`·`paths.facts`.
- `eval`: Recall@5, MRR, 시점 오염률, 최신성 정확도, 이력 완결도와 신뢰구간. 지표 정의는 `eval.py` 모듈 docstring이 기준. 요점: MRR은 상위 `eval.depth`(20)개 청크 순위로 계산, 이력 완결도는 상위 `eval.history_k`(10)개 청크가 변화 시점 날짜를 event_date 또는 본문의 완전한 날짜 표기로 덮는 비율(event_date만 쓴 값도 함께 보고). 정답 경로는 `eval.doc_prefix` 뒤로 잘라 맞춘다(볼트 이름은 설정에만). 실행 시 청크와 안 맞는 정답 문서·섹션을 경고로 출력하고 결과 meta에 남긴다. 컷오프(k, history_k, depth)마다 따로 검색한다(모드 C 변화 추적은 시간순이라 top-5가 top-20의 앞부분이 아님). 모드 `C`는 질문 해석, `CL`은 평가셋의 type·as_of 라벨로 같은 모드 C를 돌린다. C가 있으면 질문 해석 정확도(유형, as_of 일치·±7일)를 meta와 표에 남긴다. `이력(m/d 포함, 사후)`는 S5에서 결과를 본 뒤 추가한 보조 지표(공식 지표는 `history`). `DL`은 라벨로 돌린 D. D 전용 `값 정확도(D)`(사용자 결정: 자동 숫자 일치): 시점 고정·현재 값은 D 답 행의 숫자가 `answer`에 있는 비율, 변화 추적은 숫자 있는 change_points 중 ±3일 안의 D 행 값이 같은 비율. 숫자 정답이 없는 문항은 빼고, 루틴처럼 숫자 아닌 값만 답한 문항도 뺀다(`value_nonnumeric`로 셈), D가 답 못 한 문항은 0. `D 응답률`도 함께.
- 평가셋 규칙: 비공개 평가셋은 개발에 쓰지 않는다. 규칙·파라미터는 데모 평가셋과 개발용 질문으로 정해 커밋으로 고정한 뒤 한 번 돌리고, 결과를 본 뒤 바꾼 것은 "사후"로 표시하고 이전 숫자를 남긴다.
- `run`: 위 단계를 한 명령으로 묶는다.
- `demo/facts.jsonl`: 데모 볼트의 사실 관측 35행(공개, `scripts/write_demo_facts.py`가 씀). 체중은 집 체중계 아침 공복이 기본 계열, 식후·운동 직후·인바디 체중은 별도 계열.
- `demo/`: 가상 인물의 운동·식단 볼트(`vault/topics` 6, `vault/sessions` 24)와 공개 평가셋 `eval.jsonl`(4유형 각 5문항, 기준일 2025-06-30). 일부러 넣은 함정: 측정 조건이 다른 체중(식후·운동 직후), 세 번 넘게 바뀐 칼로리 한도, `created` 날짜를 가진 "현재 기준" 청크, 옛 값과 새 값이 한 청크에 있는 경우.

## 공개·비공개 경계
- 이 레포는 공개 예정이다. 들어가도 되는 것: 코드, 합성 데모 볼트와 그 공개 평가셋, 평가 숫자.
- 레포 밖 `../private` 에만 두는 것: 실제 볼트 스냅샷, 인덱스, 비공개 평가셋, 결과 원본, 유출 차단 목록, 실제 사실표(`facts/`). 레포 밖 `../models`: 모델 캐시.
- 실제 볼트의 경로·폴더명·노트 제목·본문 인용을 코드, 테스트, 주석, 커밋 메시지, 예시 설정에 쓰지 않는다. 테스트 문자열은 가상의 것을 쓴다.
- 실제 경로는 `configs/local.yaml`(gitignore)에만 쓴다. `configs/example.yaml`은 자리표시자만.
- 유출 방지 훅 `.githooks/pre-commit`: 스테이징된 파일 경로와 추가된 줄에 차단 목록 문자열이 있으면 커밋을 막는다. 차단 목록 경로는 `git config tvrag.denylist`로 지정하며(비공개 폴더의 `leak-denylist.txt`), 지정이 없거나 파일이 없으면 커밋을 막는다. 새 스냅샷을 만들면 `python scripts/build_denylist.py <스냅샷 폴더>`로 다시 만든다. 클론 후 `git config core.hooksPath .githooks`와 `git config tvrag.denylist <경로>` 필요.
- 볼트 원본은 읽기만 한다. 원격은 GitHub `0gnm0-afk/temporal-vault-rag`(공개, 열람 전용 LICENSE). push는 사용자가 요청할 때만 한다. 커밋 작성자는 GitHub noreply 주소를 쓴다(`git config user.email`).
