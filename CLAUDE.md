# temporal-vault-rag

날짜가 붙은 Markdown 볼트에서 시점을 고려해 검색하는 RAG 실습 레포. 계획·진행 상태의 단일 상태판은 상위 폴더의 `PLAN.md`이며, 이 파일은 레포 안에서 일할 때 필요한 것만 적는다.

## 명령 (Windows, 레포 루트 기준)
- 환경 활성화: `.venv\Scripts\Activate.ps1` (Python 3.12, CPU 전용 torch)
- 재설치: `py -3.12 -m venv .venv` → `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu` → `pip install -r requirements.txt` → `pip install -e . --no-deps`
- `.env` 만들기: `.env.example` 복사. 모델 캐시는 `HF_HOME`(레포 밖 `../models/hf`)만 쓴다. `SENTENCE_TRANSFORMERS_HOME`을 넣으면 sentence-transformers가 다른 캐시를 찾아 모델 로딩이 실패한다.
- 모델 받기: `python scripts/download_model.py` (bge-m3 dense 가중치만, ONNX·colbert·sparse 제외, 약 2.2GB)
- 테스트: `python -m pytest -q` / 모델 로딩 제외: `python -m pytest -q -m "not slow"` / 오프라인: `$env:HF_HUB_OFFLINE=1`
- 콘솔 한글이 깨지면 `$env:PYTHONUTF8=1` (출력 인코딩 문제일 뿐 동작에는 영향 없음)

## 모듈과 데이터 흐름 (`src/tvrag/`)
`config` → `ingest` → `chunk` → `index` → `retrieve` → `eval`, 모드 D는 `facts`가 추가로 붙는다.
- `config`: `.env`를 읽어 HF 캐시 경로를 먼저 지정하고 YAML 설정을 읽는다. HF를 쓰는 모듈보다 먼저 import한다. 상대 경로는 레포 루트 기준.
- `ingest`: 스냅샷 폴더의 Markdown → 문서(프론트매터 + 본문).
- `chunk`: 헤딩 단위 청크. 사건 시각은 날짜 헤딩 → 프론트매터 `date` → `created` 순.
- `index`: bge-m3 임베딩을 LanceDB에, kiwipiepy 형태소로 BM25 인덱스. LanceDB를 고른 이유: 파일 기반 임베디드 저장소라 서버가 없고, 메타데이터(사건 시각) 컬럼에 where 필터를 걸 수 있어 as-of 검색에 맞다.
- `retrieve`: 모드 A 벡터, B +BM25(RRF), C +시간(as-of 필터, 최신성 가중, 변화 질문은 시간순 다중 시점), D +사실표. 출력에 답한 모드와 근거 청크를 함께 낸다.
- `facts`: 이중 시간 사실표 `(대상, 속성, 값, 측정조건, 유효시작, 유효끝, 근거청크, 기록시각)`. 새 값은 이전 행의 유효끝만 채우고 지우지 않는다.
- `eval`: Recall@5, MRR, 시점 오염률, 최신성 정확도, 이력 완결도와 신뢰구간.

## 공개·비공개 경계
- 이 레포는 공개 예정이다. 들어가도 되는 것: 코드, 합성 데모 볼트와 그 공개 평가셋, 평가 숫자.
- 레포 밖 `../private` 에만 두는 것: 실제 볼트 스냅샷, 인덱스, 비공개 평가셋, 결과 원본, 유출 차단 목록. 레포 밖 `../models`: 모델 캐시.
- 실제 볼트의 경로·폴더명·노트 제목·본문 인용을 코드, 테스트, 주석, 커밋 메시지, 예시 설정에 쓰지 않는다. 테스트 문자열은 가상의 것을 쓴다.
- 실제 경로는 `configs/local.yaml`(gitignore)에만 쓴다. `configs/example.yaml`은 자리표시자만.
- 유출 방지 훅 `.githooks/pre-commit`: 스테이징된 파일 경로와 추가된 줄에 차단 목록 문자열이 있으면 커밋을 막는다. 차단 목록 경로는 `git config tvrag.denylist`로 지정하며(비공개 폴더의 `leak-denylist.txt`), 지정이 없거나 파일이 없으면 커밋을 막는다. 새 스냅샷을 만들면 `python scripts/build_denylist.py <스냅샷 폴더>`로 다시 만든다. 클론 후 `git config core.hooksPath .githooks`와 `git config tvrag.denylist <경로>` 필요.
- 볼트 원본은 읽기만 한다. push는 사용자가 결정한다.
