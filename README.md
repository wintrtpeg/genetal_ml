# genetal_ml

범용 시계열 ML 분석 도구. 데이터를 올리고 타겟만 고르면 전처리 · 파생변수 ·
모델비교 · 해석 · What-if 까지 한 흐름으로 이어집니다.

로컬 PC 에서 Python 으로 돌리고, 화면은 브라우저에 뜹니다. 폐쇄망 사내 PC 를
전제로 만들었습니다 — 웹폰트 · CDN 을 쓰지 않습니다.

도구 본체는 **[`ml_studio/`](ml_studio/)** 에 있습니다.

---

## 받아서 실행하기

**1. 내려받기**

이 페이지 위쪽 **`Code` → `Download ZIP`** 을 누르세요.
(또는 `git clone https://github.com/wintrtpeg/genetal_ml.git`)

**2. 압축 풀고 `ml_studio` 폴더로 들어가기**

```
genetal_ml-main/
└── ml_studio/      ← 여기로 들어갑니다
    ├── run.bat     ← 이걸 누릅니다
    └── ...
```

**3. 실행**

| OS | 실행 |
|---|---|
| Windows | **`run.bat` 더블클릭** |
| macOS / Linux | `./run.sh` |

첫 실행은 **3~10분** 걸립니다 — 가상환경 생성 → 패키지 설치 → 회귀 테스트 →
환경 점검 → 화면 실행까지 한 번에 합니다. 준비되면 브라우저가 자동으로 열립니다.

두 번째부터는 같은 `run.bat` 을 누르면 **몇 초 만에 화면만** 뜹니다.

끝낼 때는 그 창에서 `Ctrl+C` 를 누르거나 창을 닫으면 됩니다.

### 필요한 것

- **Python 3.10 ~ 3.12** (3.12 권장). 없으면 [python.org](https://www.python.org/downloads/)
  에서 설치하고 첫 화면의 **"Add python.exe to PATH"** 를 꼭 체크하세요.
- 패키지를 받을 인터넷 경로. 사내 프록시를 쓰면 `run.bat` 전에 한 번 지정하세요.

  ```bat
  set HTTP_PROXY=http://프록시주소:포트
  set HTTPS_PROXY=http://프록시주소:포트
  run.bat
  ```

실데이터가 없어도 됩니다 — 1단계의 **[가상 데이터]** 버튼으로 둘러보실 수 있습니다.

---

## 데이터마트(SQL) 를 쓰신다면

`ml_studio/connection.py` 를 열어 다섯 줄만 채우세요.

```python
HOST     = "dm.internal.example.com"
PORT     = 3108
DATABASE = "master"
USER     = "svc_ml"
PASSWORD = "..."
```

저장하고 `run.bat` 을 다시 실행하면, SQL 화면에서 접속 항목이 사라지고
**쿼리 입력창만** 남습니다. 호스트 · 계정 · 비밀번호는 화면에 뜨지 않습니다.

비워 두면 예전처럼 화면에서 직접 입력하는 방식으로 도니, 이 파일을 건드리지
않아도 그대로 쓰실 수 있습니다.

> 이 저장소의 `connection.py` 는 **비어 있습니다.** 값을 채운 뒤 그대로 커밋하면
> 비밀번호가 저장소에 남습니다. 쓰시는 PC 안에서만 채워 쓰세요.

---

## 다른 PC 로 옮길 때

`ml_studio` 폴더에서:

```bash
python scripts/make_dist.py
```

`dist/ml_studio_YYYYMMDD.zip` (약 1.4MB) 이 생깁니다. 그 파일 하나만 옮겨서
압축을 풀고 `run.bat` 을 누르면 됩니다.

가상환경(`.venv`) 과 지난 실행 결과(`runs/`), 그리고 `connection.py` 에 적어 둔
접속 정보는 **넣지 않습니다** — 계정이 딸려 나가는 사고를 막기 위해서입니다.

---

## 문제가 생기면

`ml_studio` 폴더의 **`report.bat`** (macOS/Linux 는 `./report.sh`) 을 실행하면
`diagnostic_report.txt` 가 생깁니다. **그 파일 하나만** 보내 주시면 됩니다.
OS · 파이썬 · 패키지 버전, 테스트 전문, 환경 점검 전문이 들어갑니다.
데이터 값 · 접속정보 · 비밀번호는 들어가지 않습니다.

---

## 더 읽을 것

| 문서 | 내용 |
|---|---|
| [`ml_studio/README.md`](ml_studio/README.md) | 전체 사용법 · 화면 흐름 · 설계상 지키는 것 |
| [`ml_studio/HANDOFF.md`](ml_studio/HANDOFF.md) | 지금 상태 · 남은 일 · 이미 당한 함정들 |
| [`ml_studio/CHANGELOG.md`](ml_studio/CHANGELOG.md) | 무엇을 왜 고쳤는지 |
