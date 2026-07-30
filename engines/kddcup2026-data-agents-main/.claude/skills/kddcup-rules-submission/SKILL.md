---
name: kddcup-rules-submission
description: 제출물 형식·전달·빈도 규칙 — Docker 이미지 네이밍 (`<team_id>:v<N>`), 아카이브 (`<team_id>_v<N>.tar.gz`, ≤10GB), Google Drive 공유, 1/일·30/Phase1 한도. "어떻게 제출해", "이미지 이름 뭐로 해", "10GB 넘으면", "Drive 공유 권한", "하루 몇 번", "Phase 1 총 몇 번" 같은 제출 절차 컴플라이언스 질문에서 트리거.
---

# Rules — Submission Format & Cadence

원천: https://dataagent.top/rules. 제출물의 형식, 전달 방법, 빈도 제한. 위반 시 그 제출 무효 또는 실격.

---

## 1. 제출물 형식 (룰 명시)

| 항목 | 값 |
|---|---|
| **이미지 이름** | `<team_id>:v<N>` |
| **아카이브 이름** | `<team_id>_v<N>.tar.gz` |
| **아카이브 사이즈** | **≤ 10 GB** |
| **빌드 명령** | `docker build -t <team_id>:v<N> .` |
| **저장 명령** | `docker save <team_id>:v<N> \| gzip > <team_id>_v<N>.tar.gz` |

**버전 번호 `<N>`:** 1부터 시작, 매 제출마다 증가. 같은 N으로 두 번 제출 금지 (운영진 트래킹 혼란).

**팀 ID:** 운영진이 2026-04-21 ~ 04-23에 할당.

---

## 2. 사이즈 한도 — 10 GB

**한도:** gzip 압축 후 tarball ≤ 10 GB.

**우리 권장:** ≤ 9 GB (1 GB 여유 — 마지막 layer 추가가 한도를 살짝 넘는 경우 방지).

**한도 초과 시:**
- 운영진이 reject할 가능성 높음
- 그날 submission이 무효 처리되면 1일 손실 (1/day 한도)

**우리 가드:** `scripts/build_submission.sh`가 빌드 후 사이즈 검증, 초과 시 fail.

**압축 압박 시:**
- `python:3.10-slim` 베이스 (full python보다 ~700MB 절약)
- `apt-get install --no-install-recommends`
- `pip install --no-cache-dir`
- `uv sync --no-dev` (테스트 deps 제외)
- 빌드 시 `rm -rf /var/lib/apt/lists/*`
- HuggingFace 모델 캐시는 `~/.cache/huggingface`에 누적 — 사용한 만큼만 빌드 시점에 다운로드 후 정리

---

## 3. 전달 메커니즘 — Google Drive

| 항목 | 룰 |
|---|---|
| **업로드** | Google Drive |
| **공유 권한** | **"Anyone with the link can view"** |
| 전달 | 운영진 이메일로 공유 링크 + 팀 ID |

**위반 예시:**
- "Restricted" 권한 → 운영진이 다운로드 못 함 → 그 제출 무효
- 팀 ID 누락 → 어느 팀 제출인지 매칭 안 됨
- 링크가 만료/삭제된 후 평가 시도 → 다운로드 실패

**권장 워크플로우:**
1. `submissions/dabench_v<N>.tar.gz` 빌드 후 Drive 업로드
2. 우클릭 → 공유 → "Anyone with the link" + Viewer
3. 링크 복사 → 이메일 본문에 링크 + 팀 ID + 버전 번호
4. 발송 후 `docs/SUBMISSION_LOG.md`에 기록 (날짜/링크/요약)

---

## 4. 빈도 제한 (룰 verbatim)

| 항목 | 값 |
|---|---|
| **하루 최대** | **1회 제출** |
| **Phase 1 A-board 총량** | **30회** (A-board 4/24 ~ 5/21 기간 한정) |
| **직렬성** | 직전 평가 완료 후 다음 제출 |

**"직렬성"의 의미:** 평가가 동시 수행 안 됨. 우리가 #N 보냈는데 아직 평가 중이면 #N+1 받지 않을 가능성. 운영진 처리 시간을 고려해 D−1에 마지막 제출하면 위험 — D−2에 마지막 제출 권장.

**Phase 1 A-board (4/24 ~ 5/21)**:
- 자동 채점 + 점진적 리더보드 피드백
- 57 task / **2시간 wall-clock**
- 매일 1회, 총 30회 제출 가능

**Phase 1 B-board (5/21 ~ 5/23)**:
- 이메일 직접 제출 (자동 채점기 사용 안 함)
- 324 task / **12시간 wall-clock**
- **1팀당 1회만** (A-board 30회 한도와 별개)
- A-board 최종 점수와 B-board 결과의 **가중 평균** (데이터 비율 기준)이 Phase 1 최종 점수
- B-board 평가는 5/22 ~ 5/26 내부 진행, 결과는 5/27 발표

**Phase 1 기간:** 2026-04-24 ~ 2026-05-23 (30일). 즉 **이론상 매일 1회씩 보내도 30회 다 못 채울 수도 있다.** 실제로는 처음 며칠은 셋업, 마지막 며칠은 직렬성 여유 — **유효 윈도우 ~25일**.

**우리 분배 (30회 중 ~7회만 실사용):** 자세한 budget은 `kddcup-strategy` §5.

---

## 5. 같은 날 두 번째 제출 시도

**룰:** 1/day. 위반 시:
- 두 번째가 무시될 가능성
- 또는 첫 번째가 무효 처리될 가능성
- 어느 쪽이든 우리 손해

**대응:** 우리는 **mock_scorer + holdout 둘 다 그린일 때만** 제출. 한쪽만 그린이면 1일 추가 검토. 둘 다 빨간불이면 절대 X.

---

## 6. 평가 진행 중 다음 제출

**룰:** 직전 평가 완료 후. 평가 시간은 12h + 큐 대기.

**시나리오:**
- 09:00 #N 제출 → 큐에서 대기 → 평가 시작 → 평가 중 우리가 #N+1 보내면 reject 가능
- 안전하게 → 다음 날까지 대기

`docs/SUBMISSION_LOG.md`에 매 제출의 (보낸 시각, 결과 받은 시각)을 기록해 패턴 학습.

---

## 7. 빌드 결정성 — `--frozen` 강제

룰 직접 명시는 아니지만 동일 결과 재현이 운영 자체에 영향:

```bash
# 빌드 명령 (Dockerfile 안)
RUN uv sync --frozen --no-dev  # uv.lock 따라 정확히 재현
```

**비결정성 회피:**
- 빌드 시점에 외부 API 호출하는 setup script 금지
- 시간 기반 로직 (timestamp가 가중치 결정) 회피
- `--frozen`이 없으면 minor 버전 drift로 동작 변할 수 있음

---

## 8. 컴플라이언스 체크리스트 (제출 직전)

- [ ] 이미지 이름이 `<team_id>:v<N>` 형식인가
- [ ] 아카이브 파일명이 `<team_id>_v<N>.tar.gz`인가
- [ ] **아카이브 사이즈 ≤ 9 GB**인가 (`du -h submissions/*.tar.gz`)
- [ ] Drive 공유 권한이 "Anyone with the link"인가 (incognito 창에서 다운로드 테스트)
- [ ] 이메일에 팀 ID + 링크 + 버전 번호 모두 포함되었는가
- [ ] 직전 제출이 평가 완료되었는가 (운영진 회신 확인)
- [ ] 오늘 다른 제출을 보내지 않았는가 (1/day)
- [ ] Phase 1 누적 제출이 30회 미만인가
- [ ] `docs/SUBMISSION_LOG.md`에 기록할 준비가 되었는가

---

## 9. 자주 깨지는 지점

1. **`docker save` 후 사이즈 폭증** — uncompressed layer가 큰 경우. `docker save | gzip -9` 보다 `docker save | xz` 압축률 높지만 호환성 문제. 검증: `docker load < <file>` 후 image 동작
2. **Drive 권한 "Restricted"로 잘못 공유** — 운영진 다운로드 실패 → 그날 제출 무효
3. **버전 번호 충돌** — 같은 N을 두 번 보내면 운영진 추적 혼란. SUBMISSION_LOG로 카운터 관리
4. **마지막 날 제출 시도** — 평가 큐 대기로 deadline 넘어갈 수 있음. D−2에 final 제출 권장
5. **로컬 docker run은 통과했는데 운영진 환경에서 fail** — 사전에 `local_eval.sh`로 운영진 환경 시뮬레이션 (`kddcup-submission`)

---

## 10. 진입점 한 줄 매핑

| 다루는 것 | 위치 |
|---|---|
| 빌드 + 사이즈 검증 | `scripts/build_submission.sh` |
| 로컬 평가 시뮬레이션 | `scripts/local_eval.sh` |
| 제출 회고 로그 | `docs/SUBMISSION_LOG.md` |
| Phase 1 일정 | `kddcup-overview` §3 |
| 제출 budget 분배 | `kddcup-strategy` §5 |
