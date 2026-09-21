# Jev Mario

![Jev Mario](assets/jev-mario-logo.svg)

실시간 에뮬레이터 상태를 TypeSafe Jev에 전달하고, Jev가 Super Mario Bros.의 다음 버튼 조합을 선택하는 연구용 컨트롤러입니다.

## 특징

- telemetry와 NES RAM을 구조화 상태로 변환
- 한 번의 Jev Choice 호출로 행동 결정
- API 응답 중에도 게임을 진행하는 realtime loop
- 지연, stale response, 실제 행동, 프레임별 상태 기록
- Windows native 및 브라우저 live monitor
- API 키와 ROM을 저장소에 포함하지 않음

## 설치

```powershell
uv venv --python 3.13.7 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
$env:TYPESAFE_API_KEY = "your-key"
```

API 키는 환경변수로만 설정하세요. `.env`는 커밋하지 않습니다.

## 실행

```powershell
.\.venv\Scripts\python.exe -m jev_mario.run --mode jev --controller realtime --decisions 120
```

실행 결과는 `runs/<timestamp>-realtime/`에 기록됩니다.

### Windows live monitor

```powershell
.\.venv\Scripts\python.exe -m jev_mario.live_window runs\<timestamp>-realtime
```

게임 화면, Jev 선택, 실제 행동, 위치, API 지연, 점프 상태를 표시합니다.

### 브라우저 live monitor

```powershell
.\.venv\Scripts\python.exe -m jev_mario.live_ui runs\<timestamp>-realtime
```

브라우저에서 `http://127.0.0.1:8765/`를 엽니다.

## 기록 파일

- `decisions.jsonl`: Jev 요청·응답·선택 행동·결과
- `frames.jsonl`: 프레임별 행동과 위치
- `perception.jsonl`: 프레임별 관측
- `api.jsonl`: 호출 지연과 사용량 메타데이터
- `summary.json`: 실행 요약
- `index.html`: 리플레이 보고서

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

API 키, `.env`, 개인 경로, 실행 로그, ROM 파일을 Git에 추가하지 마세요. Super Mario Bros. ROM은 이 저장소에 포함하지 않습니다.
