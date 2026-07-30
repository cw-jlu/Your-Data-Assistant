---
name: kddcup-rules-model
description: 평가 시점 LLM 사용 규칙 — qwen3.5-35b-a3b 강제, 다른 LLM 메인 솔버 금지, 보조 모델(임베딩·검색) 허용 범위. "다른 LLM 써도 돼?", "보조 모델 뭐 쓸 수 있어", "CPU에서 작은 모델 돌려도 돼", "fine-tune해도 돼", "임베딩 어떻게 써" 같은 모델 컴플라이언스 질문에서 트리거.
---

# Rules — Model Usage Policy

원천: https://dataagent.top/rules. 어떤 LLM을 어디서 어떻게 쓸 수 있는가에 대한 룰. 위반 = 즉시 실격.

---

## 1. 핵심 강제 — Qwen3.5-35B-A3B

룰 명시: "Qwen3.5-35B-A3B (Uniformly Deployed by Organizers)"

| 항목 | 값 |
|---|---|
| **평가 시 메인 솔버** | `qwen3.5-35b-a3b` **만** 허용 |
| 호스팅 | 운영진이 일괄 배포 |
| 접근 | `MODEL_API_URL` (OpenAI Chat Completions 호환) |
| 인증 | `MODEL_API_KEY = "EMPTY"` (운영진 명시 verbatim) |
| 식별자 | `MODEL_NAME = "qwen3.5-35b-a3b"` |

**운영진 공식 vLLM 배포 spec (2026-05 룰 페이지 명시):**

```bash
vllm serve <model_path> \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --served-model-name qwen3.5-35b-a3b \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
```

- **Max context: 262144 토큰** (~256K) — Hard/Extreme의 긴 context를 처리 가능
- **Reasoning parser**: qwen3 (`<think>` 태그 분리)
- **Tool calling**: `qwen3_coder` parser로 자동 tool 호출 지원
- **Seed 1024 고정** — endpoint 자체가 결정적이지 않을 수도 있음 (우리는 per-request seed 사용)

**메인 솔버(primary task-solving model)란:** 자연어 질문을 읽고 plan/reasoning/answer를 생성하는 LLM. 우리 ReAct 루프의 model adapter (`OpenAIModelAdapter`)가 호출하는 것이 곧 메인 솔버.

---

## 2. 개발 vs 평가 — 자유도 차이

| 단계 | 메인 솔버 |
|---|---|
| **개발** | 어떤 LLM이든 자유 (GPT-4o, Claude, Gemini, 오픈소스 등) |
| **평가** | `qwen3.5-35b-a3b`만 |

**전환 메커니즘:** 환경변수 오버레이.
- 개발 시: `configs/local.yaml`에 우리 vLLM/SGLang endpoint
- 평가 시: 운영진이 `MODEL_API_URL` 등 주입 → `configs/eval.yaml`의 빈 필드를 env가 채움

**원칙:** "로컬 개발 모델 = 평가 모델"로 맞추는 게 가장 큰 무기. GPT-4o로 프롬프트 튜닝하면 평가 시 무용. 우리는 회사 GPU에 `Qwen3-30B-A3B-Instruct-2507`을 vLLM으로 셀프호스팅.

---

## 3. 명시적 금지 — Prohibited Models

룰 verbatim:

> "No alternative LLM may serve as 'the primary task-solving model' during evaluation, including 'CPU-executed local inference of alternative LLMs.'"

| 행위 | 룰 |
|---|---|
| 다른 LLM을 메인 솔버로 사용 | ❌ 실격 |
| **컨테이너 내부에 작은 LLM(7B 등)을 CPU로 띄워 메인 솔버로 사용** | ❌ 명시적 금지 |
| 다른 LLM API를 컨테이너에서 호출 | ❌ 네트워크 격리로 자동 차단 + 룰 위반 |
| 평가 전 다른 LLM으로 답을 만들어 답안만 컨테이너에 배포 (offline 답안 캐시) | ❌ 명시적 금지 (probing/cache 카테고리) |

**즉:** 평가 시 우리 컨테이너 안에서 **`MODEL_API_URL`로 가는 호출만이 추론을 만들 수 있다.**

---

## 4. 명시적 허용 — Auxiliary Models

룰 verbatim:

> "Auxiliary models (embeddings, retrieval) are permitted within hardware budgets."

| 용도 | 허용 |
|---|---|
| 임베딩 (sentence-transformers, BGE 등) | ✅ |
| Reranker (cross-encoder) | ✅ |
| 텍스트 분류기 (소형) | ✅ (메인 솔버가 아닌 한) |
| OCR | ✅ (CPU에서 — pytesseract 등) |
| 토크나이저 | ✅ |

**제약:**
- 16 vCPU / 64GB RAM / GPU 없음 (`kddcup-rules-compute`)
- 모든 모델 가중치는 Docker 이미지에 포함 (네트워크 차단)
- ≤ 10 GB tarball 한도 안에 들어가야 함 (`kddcup-rules-submission`)

**경계 케이스 — "보조 모델"의 정의:**
- ✅ 임베딩으로 candidate row를 추리고 → qwen이 최종 답
- ❌ 작은 LLM이 plan을 짜고 → qwen이 실행만 (이건 "메인 솔버"가 작은 LLM이 됨)
- ✅ 작은 분류기가 "이 컬럼이 numeric인지" 판단 → qwen이 답 생성
- ❌ 작은 LLM이 답을 생성하고 qwen이 검증만 (메인 솔버 분리 위반)

**원칙:** 자연어 질문을 받아서 reasoning을 만드는 게 메인 솔버. 그 외 모든 모델은 보조.

---

## 5. Fine-tuning

룰 명시는 없지만 **사실상 무용**:
- 평가 시 우리는 운영진이 호스팅한 qwen 가중치만 쓴다
- 우리가 fine-tune한 가중치는 평가 컨테이너에서 로드할 수 없다 (네트워크 차단 + 운영진 endpoint만)
- LoRA/adapter도 운영진 endpoint가 받지 않으므로 무용

**대응:** fine-tune 대신 **프롬프트 엔지니어링 + few-shot + plan-then-execute**에 집중. 자세한 전략은 `kddcup-strategy`.

---

## 6. 하드코드 금지 (model 측 측면)

룰 명시: "Teams may use any LLM during development but must read endpoint details from environment variables, not hardcode them."

**우리 코드에서 만지는 곳:**
- `configs/eval.yaml`: `model: ""`, `api_base: ""`, `api_key: ""` (빈 문자열 유지)
- `config.py:load_app_config`: env > YAML 우선순위 (이미 와이어드)
- `agents/model.py:OpenAIModelAdapter`: 인자로 받은 `api_base`/`api_key` 사용 (하드코드 X)

**위반 예시:**
```python
# 절대 금지
client = OpenAI(api_key="sk-real-key", base_url="https://my-server")
```

```yaml
# 절대 금지 (configs/eval.yaml)
agent:
  api_base: "http://my-vllm:8000/v1"   # ← 평가 시점에도 이 값으로 시도하게 됨
```

---

## 7. 우리 환경 vs 평가 환경 — 모델 측 차이

| 항목 | 우리 (DGX vLLM) | 운영진 평가 |
|---|---|---|
| 모델 | `Qwen3-30B-A3B-Instruct-2507` (alias `qwen3.5-35b-a3b`) | 진짜 `qwen3.5-35b-a3b` |
| Max context | 32K (YaRN 미적용) | 미상 — Discord 질의 |
| JSON-mode (`response_format`) | 지원 (vLLM) | 미상 — 초기화 시 probe |
| latency | LAN | 운영진 인프라 |
| throughput | 우리 GPU 한도 | 운영진 한도 |

**가장 큰 미지수 (Phase 0에 Discord로 질의 권장):**
1. JSON-mode 지원 여부
2. Max context window
3. 동시 요청 한도 (rate limit)

자세한 endpoint capability 점검은 `docs/qwen_endpoint_capabilities.md`.

---

## 8. 컴플라이언스 체크리스트

- [ ] `configs/eval.yaml`의 `model`/`api_base`/`api_key`가 모두 빈 문자열인가
- [ ] 코드 내 hardcoded URL/key가 없는가 (`grep -r "api_key\s*=" src/` / `grep -r "api_base\s*=" src/`)
- [ ] 컨테이너 내부에 추가 LLM 가중치가 없는가 (`docker history` / `du -sh` 검토)
- [ ] 보조 모델이 있다면 메인 솔버 역할을 하지 않는가 (코드 흐름 확인 — 자연어 질문에 응답하는 마지막 호출이 qwen인가)
- [ ] fine-tune된 모델 / adapter / LoRA 파일이 image에 포함되지 않는가 (의도치 않은 dev artifact)

---

## 9. 자주 깨지는 지점

1. **dev에서 `OPENAI_API_KEY` env로 GPT-4o 쓰던 코드가 컨테이너에 묻어가서 평가 시 GPT 호출 시도** → 네트워크 차단으로 fail + 코드 흔적 발견 시 룰 위반
2. **Hugging Face 모델 자동 다운로드 코드** — 평가 시 네트워크 차단으로 fail. 빌드 시점에 `HF_HUB_OFFLINE=1` + 가중치 사전 다운로드
3. **임베딩 모델이 메인 솔버처럼 작동** — "이 LLM이 자연어 질문에 답을 만드는가"로 self-check
4. **YAML에 dev API key 남아있음** — `eval.yaml` 외 다른 config가 잘못 마운트되거나 빌드되면 위반
