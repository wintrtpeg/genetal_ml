# 인수인계 — 대화가 끊겼을 때 이것부터 읽으세요

이 파일은 **새 대화 세션에 이 프로젝트를 그대로 넘기기 위한 것**입니다.
`ml_studio.zip` 을 새 대화에 올리고 "HANDOFF.md 부터 읽고 이어서 진행해" 라고 하면
됩니다.

---

## 1. 이게 뭔가

도메인에 묶이지 않는 범용 시계열 ML 도구. 제조/플랜트 센서 데이터를 올리고
타겟만 고르면 전처리 → 파생변수 → 모델비교 → 해석 → What-if 까지 이어집니다.
로컬 VS Code + Python, 화면은 Streamlit.

전체 사용법은 `README.md`, 지금까지 무엇을 왜 고쳤는지는 `CHANGELOG.md` (11차, 1,200줄).

---

## 2. 지금 상태 — 한 줄로

**기능은 전부 구현됐고 테스트 360건이 통과합니다. 남은 것은 "실제 환경에서
눈으로 보는 검증" 뿐입니다.**

| 층위 | 상태 |
|---|---|
| 원래 요청 12가지 | 전부 구현 |
| 로드맵 16개 + 추가 요청 | 전부 구현 |
| 자동 검증 | **360건 통과, 0 실패** |
| 화면 실행 검증 | 대역으로 10개 화면 × 상태 40여 조합 실행 |
| **진짜 브라우저 렌더링** | **미확인** — 레이아웃·색·체감속도. ← 지금 여기 |
| **진짜 plotly/shap/부스팅** | **확인됨** (12차, 집 PC) — SHAP 3경로·부스팅 3종·차트 22종 통과 |
| **SQream 실접속** | **미확인** — 계정 필요 |
| **실데이터** | **미확인** — 가상 데이터로만 검증 |

---

## 3. 절대 깨면 안 되는 것 (보존 원칙)

이걸 어기는 수정은 되돌려야 합니다. `tests/` 가 전부 회귀 테스트로 잡습니다.

1. **`core/` 는 streamlit 을 import 하지 않는다** — Dataiku 이식성 전제
2. **Y 파생 피처 차단** — `assert_no_target_derived`, 우회 경로 없음
3. **파생변수는 backward-only** — 중앙정렬 rolling·음수 shift 금지
4. **전처리는 sklearn Pipeline 안에서** — 사전 fit 금지
5. **웹폰트 금지** — 폐쇄망에서 CDN 차단됨
6. **core 에 도메인 하드코딩 금지** — 설비·공정·태그 이름
7. **기존 테스트는 계속 통과해야 한다**
8. **D7 — 파생변수 생성에 LLM 을 쓰지 않는다** (데이터 반출 제한). 규칙 기반만
9. **SQL 은 SELECT/WITH 만**, 접속정보는 `runs/` 에 저장하지 않고 마스킹만

추가로 **Final Unseen 은 분할당 1회만** 열립니다 (11차에 코드 수준으로 강제).
`UnseenGuard` 는 **분할에 딸린 것**이지 학습 실행에 딸린 것이 아닙니다 —
`app/state.py` 의 `select_out` 에 있고 `train_out` 에는 없어야 합니다.

10. **추천은 추천이지 강제가 아니다** (16차) — `core/advisor.py` 가 데이터를 보고
    전처리·lag·rolling·선별상한을 추천하지만, 사용자가 다른 값을 고르는 것을
    **막지 않습니다.** 다르게 고르면 "추천 X 대신 Y 로 진행합니다" 라고 알려만
    줍니다. 현장 판단이 통계보다 옳은 경우가 많습니다.
11. **추천에는 반드시 사유와 근거가 붙는다** — 값만 주면 검증할 수 없는 강요가
    됩니다. `Advice(value, reason, detail, confidence, notes)` 다섯 가지가 한 벌입니다.
    사유 문장은 접지 않습니다 (접으면 아무도 안 폅니다). 근거 표만 접습니다.
12. **물리적 한계는 통계를 이긴다** — `PhysicalLimits` 를 넘는 lag·rolling 후보는
    추천에서도 선택지에서도 **제거**합니다 (비활성화가 아니라 제거).

---

## 4. 지금 해야 할 일 (우선순위)

### 4-1. 집 PC — 화면을 눈으로 본다  ← **여기부터**

지금까지 **아무도 이 UI 를 브라우저에서 본 적이 없습니다.** 대역으로 "터지지
않는다" 까지는 확인했지만 레이아웃·문구·체감속도는 못 봅니다.

```
run.bat          → 첫 실행: 설치 · 테스트 · 환경점검 · 화면 실행 (3~10분)
run.bat          → 두 번째부터: 화면만 (몇 초). 브라우저가 자동으로 열립니다
run.bat --full   → 설치와 점검을 다시 하고 싶을 때
```

보고 알려줄 것: 겹치거나 잘린 곳 / 굼뜬 조작 / 빈 차트 / 이해 안 되는 문구 /
다음에 뭘 눌러야 할지 모르겠는 지점.

### 4-2. `scripts/verify_env.py` 결과

설치가 끝난 PC 에서만 확인되는 28건 — 차트 21종, SHAP 3경로, 부스팅 3종,
SQream 엔진, HTML 리포트. `run.bat` 첫 실행에 이미 포함돼 있습니다.

### 4-3. 회사 PC — SQream 실접속 + 실데이터

집에서 화면 문제를 털어낸 뒤에 합니다. 순서를 바꾸면 화면 버그와 데이터 문제가
섞여 원인 찾기가 어려워집니다.

### 4-4. 보류 중 (사용자가 우선순위에서 뺌)

**분류(classification) 화면 경로.** `core/` 는 이미 지원합니다 — `detect_task`,
F1·ROC_AUC, 분류 모델 zoo, 앙상블. 하지만 5·6단계 화면(실측대비 라인차트,
SHAP dependence)이 회귀 기준으로 만들어져 있습니다.

---

## 5. 문제가 생기면

```
report.bat       (Windows)      →  diagnostic_report.txt 생성
./report.sh      (macOS/Linux)
```

그 파일 하나만 대화창에 첨부하면 됩니다. OS·파이썬·패키지 버전, 회귀 테스트
전문, 환경 점검 전문, 축소 end-to-end 가 다 들어갑니다. 맨 앞에 통과·실패 요약.
데이터 값·접속정보·비밀번호는 들어가지 않습니다.

---

## 6. 구조 — 어디를 고쳐야 하나

```
core/          streamlit 을 모르는 순수 로직 (이식 가능해야 함)
  datasource   CSV · SQLAlchemy(SQream 포함) · 시계열 정규화
  profiling    컬럼 품질 진단
  preprocess   Pipeline 조립 (임퓨터·스케일러·인코더)
  features     파생 생성 · 선별(select_core) · FoldSelector · 검토표
  validation   3분할 · gap · 누수 점검표 · rolling 윈도우
  models       모델 zoo · task 판정
  train        병렬 학습 · 챔피언 · UnseenGuard · 백테스트
  ensemble     자체 OOF 스태킹 (sklearn 것은 TimeSeriesSplit 에서 못 씀)
  tuning       nested CV 탐색
  explain      SHAP 3경로 + 순열 중요도 대체
  whatif       시나리오 · PDP/ICE
  diagnostics  잔차 · drift · 이상점 · 자기상관
  plots        차트 22종 (다운샘플링 · WebGL 임계)
  persist      run 저장 · manifest · Champion-Challenger
  report       단독 HTML 리포트
  pipeline     Auto 모드 (run_auto)
  config       설정 직렬화
  advisor      데이터를 보고 설정을 추천 (값·사유·근거) + PhysicalLimits
  housekeeping 저장공간 정리 (보관 지정 · 용량 예산 · runs 밖 금지)

app/           Streamlit 화면 (core 를 호출만 함)
  main         진입점 · 사이드바 · 상태바
  state        DEFAULTS · invalidate 체인 · ready/guard
  nav          단계 이동 · blocker · advice
  theme        디자인 토큰 · csv_download
  advice_ui    추천을 화면에 붙이는 공통 부품 (why · deviation · limits_form)
  views/       10개 화면

tests/         360건
  fake_streamlit / fake_plotly   대역 — 화면을 실제로 실행하기 위한 것
  test_view_render               화면 10개 × 상태 40여 조합
  test_runtime_guards            실행 오류 · 성능 · 누수 접근권
  test_leakage / test_split_leakage   누수 방지 장치
  run_tests                      pytest 없이도 도는 러너

scripts/
  setup          첫 실행 (run.bat 이 부름). 두 번째부터는 launch 로 바로 감
  launch         화면 실행 + 포트 선택 + 브라우저 자동 열기
  verify_env     설치된 PC 에서만 확인되는 것들
  collect_report diagnostic_report.txt 생성 (report.bat 이 부름)
  smoke_test     전체 end-to-end
  quick_check    축소 end-to-end
  make_demo_data 가상 데이터
```

---

## 7. 함정 — 이미 당한 것들

새 세션이 같은 데 빠지지 않게 적어 둡니다. 자세한 경위는 CHANGELOG 참조.

| 함정 | 요약 |
|---|---|
| `sort_leaderboard` 중복 rank | `drop(columns=["rank"], errors="ignore")` 로 idempotent |
| family 이름 충돌 | RandomForest 의 sklearn family 가 "ensemble" — 결합모델로 오판 |
| gap 점검 죽은 코드 | `gap >= 0` 은 항상 참. `gap >= max_lookback` 이어야 함 |
| 테스트 러너 fixture | pytest 설치 환경에서만 깨졌음. 표식이 3가지 |
| 러너 skip 이 가짜 | `lambda: None` 이라 skip 뒤 코드가 실행됐음 |
| cp949 에 em dash 없음 | 출력을 파일로 넘기면 UnicodeEncodeError. `_enable_utf8()` |
| `ready()` 가 대리 조건 | champion 만 보고 X·y·split 은 안 봤음 |
| Unseen 접근권 부활 | 가드를 학습 실행에서 만들어서 재학습마다 초기화됐음 |
| 검사기 자체의 구멍 | sprintf 정규식이 `%.2%` 를 통과시킴. 검사기도 테스트해야 함 |
| 추천을 최악값으로 판단 | 센서 하나가 60% 비었다고 전체 대치 방식을 바꿈. 대표값으로 봐야 함 |
| 거의 상수인 태그 | σ=1e-6 이 절대 임계를 통과해 "1,800만 배 차이" 사유가 나옴. 변동계수로 |
| lag 를 절대 개선치로 | 0.993→0.999 를 "+0.006, 하찮음" 으로 봐서 진짜 지연을 놓침. **오차 감소율**로 |
| MAD=0 이면 이상값이 숨음 | 값이 고정된 센서 + 스파이크 1개 → robust z 가 0. 표준편차로 물러서야 함 |
| 추천이 위젯에 안 닿음 | 추천을 계산만 하고 기본값으로 안 쓰면 화면은 멀쩡히 그려짐. 대역이 기본값을 기록해야 잡힘 |
| plotly `titlefont` | 5.x 에서 제거됨. `title=dict(font=...)` |
| 폴드 선별의 MI 낭비 | `top_k` 없으면 MI 는 안 쓰임. 50만행 기준 61분 → 36초 |
| quick_check 의 gap | `max(lags)` 는 rolling 창을 빼먹음. `warmup_rows(cfg)` 를 쓸 것 |
| `--quick` 이 결함을 가림 | 시간 아끼려던 옵션이 유일하게 그 결함을 잡는 단계를 건너뜀 |
| 점검 스크립트의 가짜 데이터 | 제품이 내는 컬럼과 다르면 가짜 경보가 난다 |
| 드라이버 없음 ≠ 결함 | SQLAlchemy 는 `NoSuchModuleError` 를 던진다. 건너뜀으로 분류 |
| `SystemExit(문자열)` | `.code` 가 문자열이라 `int()` 가 터진다 |

---

## 8. 이 환경(클라우드 컨테이너)의 제약

작업하는 쪽 컨테이너는 **패키지 저장소가 막혀 있습니다** (`pypi.org` 403,
미러·github raw 전부 차단, 로컬 wheel 없음). 그래서 이쪽에는
**streamlit · plotly · shap · 부스팅 3종 · SQLAlchemy 가 없습니다.**

- 있는 것: pandas, numpy, scikit-learn, scipy, joblib, PyYAML
- 그래서 누수 검증·학습·스모크 테스트는 전부 여기서 돌아갑니다
- 화면·차트는 `tests/fake_streamlit.py` · `tests/fake_plotly.py` 대역으로 실행합니다

**사용자 PC 는 이 제약이 없습니다.** 회사 PC 에서도 `requirements-core.txt` 는
정상 설치됐습니다 (streamlit·plotly 포함). 이 컨테이너의 제약을 사용자 환경의
제약으로 착각하지 마세요 — 한 번 그렇게 잘못 말한 적이 있습니다.
