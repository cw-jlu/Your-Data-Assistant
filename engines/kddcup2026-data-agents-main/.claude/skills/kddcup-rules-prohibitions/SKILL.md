---
name: kddcup-rules-prohibitions
description: 룰 페이지에 명시된 금지 행위 — 외부 인터넷 접근, 비-Qwen 메인 솔버, /input 수정, env 변조, 평가 인프라 probing, 이미지 공유, 다른 팀 명의 제출. "이거 해도 돼?", "실격 사유", "프로빙", "팀 간 공유", "환경 변수 바꿔도 돼" 같은 컴플라이언스 합법성 판단 질문에서 트리거. 의심스러운 행위는 무조건 이 스킬로 점검.
---

# Rules — Prohibited Behaviors & Disqualification Triggers

원천: https://dataagent.top/rules. **룰 위반 = 즉시 실격**. 모든 의심스러운 최적화는 이 스킬로 1차 검증.

---

## 1. 명시적 금지 행위 (Critical Prohibitions, 룰 verbatim)

룰 페이지의 "Critical Prohibitions" 섹션 그대로:

| # | 금지 행위 |
|---|---|
| 1 | **외부 인터넷 접근**, 또는 `MODEL_API_URL` 우회해서 다른 LLM 서비스 호출 |
| 2 | **비-Qwen 모델을 메인 솔버로 컨테이너 안에서 실행** |
| 3 | `/input` 디렉토리 수정 또는 주입된 환경변수 파괴 |
| 4 | **Docker 이미지를 팀 간 공유**, 또는 다른 팀 명의로 제출 |
| 5 | **평가 인프라 probing/공격** |

각각 단순한 "권장사항"이 아니라 **실격 사유**.

---

## 2. #1 — 외부 인터넷 우회

**규칙:** `MODEL_API_URL` 외 모든 외부 통신 금지.

**위반 패턴:**
- `requests.get("https://api.openai.com/...")` — 다른 LLM 호출
- `huggingface_hub.snapshot_download(...)` — 런타임 모델 다운로드
- `pip install ...` 실행 — 패키지 다운로드
- `urllib.request.urlopen(...)` — 어떤 외부 URL이든
- DNS 쿼리 자체가 막혔을 가능성 — `socket.gethostbyname("google.com")`도 fail

**자기 진단:**
```bash
# 빌드된 컨테이너 실행 후 외부 호출 시도해서 차단되는지 확인
docker run --rm --network=eval_net <image> \
  uv run python -c "import urllib.request; urllib.request.urlopen('https://google.com', timeout=2)"
# expected: 즉시 실패
```

**합법:** `MODEL_API_URL`에 대한 호출만. 그 endpoint가 우리 dev에서는 LAN vLLM, 평가 시에는 운영진 endpoint.

---

## 3. #2 — 비-Qwen 메인 솔버

**규칙:** "primary task-solving model"은 `qwen3.5-35b-a3b`만. CPU에서 작은 LLM 띄워서 메인 솔버로 쓰는 것도 명시적 금지.

**메인 솔버 정의:** 자연어 질문을 받아서 plan/reasoning/answer를 생성하는 LLM. ReAct 루프의 model adapter가 호출하는 것.

**위반 패턴:**
- 컨테이너 안에 Llama-3-8B 가중치 포함 → CPU로 띄워서 실제 추론 → 룰 위반
- 작은 LLM이 "쉬운 태스크"는 직접 답을 만들고 → "어려운 태스크"만 qwen에 위임 → 위반 (메인 솔버가 case별로 분기됨)
- 작은 LLM이 plan을 짜고 → qwen이 step별 실행만 → 위반 (reasoning이 작은 LLM에서 발생)

**합법 (보조 모델):**
- 임베딩 모델로 candidate row 추리기 → qwen이 최종 답
- Cross-encoder로 reranker → qwen이 답 생성
- 분류기가 metadata 판단 → qwen이 답

자세한 보조 모델 정의는 `kddcup-rules-model` §4.

---

## 4. #3 — /input 수정 / env 변조

**`/input` 측:**
- read-only mount이지만 우리 코드에서 명시적으로 `chmod`, 우회적 마운트, hard-link 등 시도 = 위반
- 임시 파일을 `/input` 안에 만들면 안 됨 (RO이지만 의도 자체가 위반)

**환경변수 측:**
```python
# 절대 금지
del os.environ["MODEL_API_KEY"]
os.environ["MODEL_API_URL"] = "http://my-server"  # 운영진 값 덮어쓰기
```

**예외:** 우리가 set하지 않은 env var (예: `DABENCH_LAMBDA`)를 우리가 만들어 set하는 건 OK. 운영진이 주입한 3개(`MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME`)만 변조 금지.

---

## 5. #4 — 팀 간 이미지 공유 / 다른 팀 명의 제출

**금지:**
- 팀 A가 팀 B의 docker image를 받아서 자기 명의로 제출
- 같은 코드베이스를 두 팀이 다른 v 버전으로 위장 제출
- 팀원이 두 팀에 걸쳐 활동

**시사점:** 우리는 single-team 정책 — 팀 외부에 코드/이미지 공유 X. 오픈소스 공개도 Phase 1 종료 전에는 금지 권장.

**자기 진단:** GitHub repo가 public이면 다른 팀이 fork해서 제출 시도 가능. **Phase 1 기간(2026-04-24 ~ 05-23) 동안 private 유지** 권장.

---

## 6. #5 — 평가 인프라 Probing/공격

**위반 예시:**
- 평가 노드의 호스트명/IP/네트워크 정보 수집 (`ifconfig`, `cat /etc/hosts`, `traceroute`)
- 마운트 외 디렉토리 접근 시도 (`/etc`, `/proc/1`)
- 다른 팀의 컨테이너 추측 (`docker ps`는 어차피 안 되지만)
- `MODEL_API_URL`을 다른 endpoint에 reverse-proxy로 forward
- 의도적으로 minimal/maximal prediction.csv를 보내서 gold schema 역추정 (adversarial schema probing)

**adversarial schema probing 특별 주의:**
- 1열 빈 csv → recall 높이는 행위 측정
- 100컬럼 가짜 csv → extra penalty 측정
- 이런 식으로 λ나 gold 구조를 역추정하는 시도 = 룰 위반 가능성

운영진은 위 행위를 detect할 가능성이 있다 (제출별 score 패턴 비교). **30회 예산을 진짜 개선에만 사용** — 자세한 분배는 `kddcup-strategy` §5.

---

## 7. 룰에 없지만 회색지대인 행위

| 행위 | 위험도 | 의견 |
|---|---|---|
| Phase 1 dataset의 task_id를 outside source에서 leak받아 사전 답안 캐싱 | **즉시 실격** | 명시적 금지는 아니지만 #1 (외부 데이터)·#5 (probing) 양쪽에 걸침 |
| 같은 팀이 여러 Discord 계정으로 운영진에 같은 질문 반복 | 낮음 | 룰 외, 그러나 신뢰 손해 |
| Test-set tasks의 metadata만 외부에서 입수 (예: 운영진 발표 슬라이드) | 회색 | leakage 의심 |
| 우리 컨테이너가 의도치 않게 telemetry 라이브러리(예: posthog) 호출 | **위반** | 외부 인터넷 접근에 해당. 빌드 시 명시적으로 끄기 |
| 운영진 endpoint의 동작 패턴(rate limit, retry semantics) 학습 | **회색** | 정상 사용 범위면 OK. 의도적 abuse는 probing 의심 |

**원칙:** "이게 룰 위반인가?"가 의심스러우면 **하지 말 것**. 30회 중 한 번을 잃는 비용 < 실격 비용.

---

## 8. 우리 코드에서 자기검증

```bash
# 1. 외부 호출 패턴 grep
grep -rE "requests\.|urllib|httpx|aiohttp" src/

# 2. 다른 LLM API 흔적
grep -rE "openai\.com|anthropic|cohere|together\.ai|huggingface" src/

# 3. /input 수정 시도
grep -rE "open\(.*['\"]/(input|/input)" src/   # 절대경로
grep -rE "shutil\.copy.*input" src/

# 4. env var 변조
grep -rE "os\.environ\[.MODEL_(API|NAME)" src/   # set 시도

# 5. 인프라 probing 흔적
grep -rE "subprocess.*(ifconfig|hostname|traceroute)" src/
```

빌드 직전에 위 5개 grep 모두 결과 없는지 확인.

---

## 9. 컴플라이언스 체크리스트 (제출 전 마지막)

- [ ] 컨테이너 안에서 외부 인터넷 호출 시도 코드가 없는가
- [ ] 다른 LLM API 호출 코드가 없는가 (OpenAI, Anthropic, HuggingFace 등)
- [ ] 메인 솔버는 오직 `MODEL_API_URL` 호출만인가
- [ ] 보조 모델이 메인 솔버처럼 추론을 만들지 않는가
- [ ] `/input` 수정 코드가 없는가
- [ ] `MODEL_API_URL`/`MODEL_API_KEY`/`MODEL_NAME` env 변조 코드가 없는가
- [ ] 평가 노드 정보 수집 시도가 없는가
- [ ] 컨테이너가 다른 팀과 공유되지 않았는가 (private repo 유지)
- [ ] adversarial probing submission이 없는가 (30회 예산 모두 진짜 개선용)
- [ ] Telemetry/analytics 라이브러리가 자동 호출되지 않는가 (posthog, sentry 등)

---

## 10. 자주 깨지는 지점

1. **dev에서 `requests` 라이브러리로 GPT-4o 쓰던 코드가 컨테이너에 묻어감** — 평가 시 차단되어 fail + 룰 위반 의심
2. **HuggingFace 모델 자동 다운로드 (transformers 기본 동작)** — 런타임에 외부 호출. 빌드 시점에 `HF_HUB_OFFLINE=1` + 사전 다운로드
3. **OpenAI Python SDK가 telemetry 호출** — `OPENAI_LOG=...` 같은 env에 따라. 우리는 vLLM이라 OK이지만 검증
4. **Sentry/posthog/datadog SDK** — 의존성에 따라 자동 import + 호출. `pip list | grep -iE "sentry|posthog|telemetry"`
5. **GitHub repo public 상태로 Phase 1 시작** — 다른 팀 fork 위험. private로 전환

---

## 11. 진입점 한 줄 매핑

| 다루는 것 | 위치 |
|---|---|
| 룰 페이지 원본 | https://dataagent.top/rules |
| 모델 측면 (메인 vs 보조) | `kddcup-rules-model` |
| 런타임 측면 (외부 호출 차단) | `kddcup-rules-runtime` |
| 제출 측면 (이미지 공유) | `kddcup-rules-submission` |
| 30회 예산 분배 (probing 회피) | `kddcup-strategy` §5 |
