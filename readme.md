# 🚀 RisuMemoryBackend: The Self-Constructing AI Ego (Memory Core)

---

### 📘 이게뭐시랭

🙏본 프로젝트는 risuAi의 **장기기억(Long-Term Memory)** 관리 방식을 제미나이 2.5로 긴빠이쳐와서 파이썬 백엔드 라이브러리로 구현해 보았습니다. 🙏[RisuAI](https://github.com/kwaroran/RisuAI) 개발팀에게 감사를 표하며, 이 프로젝트는 복잡한 대화내역 알아서 요약시키고 효율적으로 저장시키고 벡터로 저장, 요약, 검색하여 AI의 맥락 이해 능력과 일관성을 향상시키는 데 목적이 있습니다. 

- 현제 버전은 독립적인 API 백엔드로 작동할 수 있으며, 추후 다른 서비스와 통합될 수 잇습니다.


---

## 🚀 주요 기능 및 모듈

### 1. HypaMemoryV3

고급 벡터 검색 기반 메모리 관리 모듈입니다.

-   **기능:**
    -   대화 요약 (Summarization): Gemini API 또는 설정된 모델을 사용하여 대화 내용을 요약합니다.
    -   임베딩 생성 (Embedding Generation): 요약된 텍스트의 벡터 표현을 생성하여 저장합니다.
    -   유사도 검색 (Similarity Search): 현재 대화 내용을 기반으로 가장 관련성 높은 과거 메모리를 검색합니다.
    -   기억 선별 (Memory Prioritization):
        -   `is_important` 플래그: 중요하다고 표시된 메모리는 항상 컨텍스트에 포함됩니다.
        -   유사도 (Similarity): 최근 대화와의 관련성 기반.
        -   최근성 (Recency): 대화 기록 순서 기반.
        -   토큰 제한 관리: 설정된 `memory_tokens_ratio`에 맞춰 메모리 사용량을 조절합니다.
-   **주요 설정:** `HypaV3Settings` (모델, 프롬프트, 토큰 비율 등)

### 2. SupaMemory

대규모 대화 기록을 효율적으로 관리하기 위한 메모리 압축 및 요약 모듈입니다.

-   **기능:**
    -   대화 내용 압축 (Summarization): Gemini API 또는 설정된 모델을 사용하여 대화의 특정 청크를 요약합니다.
    -   토큰 효율성 (Token Efficiency): 최대 컨텍스트 길이를 초과할 때, 오래된 대화부터 요약하여 토큰 수를 줄입니다.
    -   시스템 메시지 삽입: 요약된 내용은 `system` 메시지 형식으로 변환되어 최종 컨텍스트에 포함됩니다.

### 3. Tokenizer

텍스트의 토큰 수를 계산하는 유틸리티 모듈입니다.

-   **기능:**
    -   `tiktoken` 라이브러리 기반으로 OpenAI 모델(GPT-4, GPT-3.5 등)의 토큰 수를 정확하게 계산합니다.
    -   단일 메시지 (`count_chat_tokens`) 및 전체 대화 기록 (`count_chat_history_tokens`)의 토큰 수를 계산합니다.

---

## 🚀 API 백엔드 실행

이 백엔드는 FastAPI를 통해 API 서버로 실행될 수 있습니다.

### 1. 사전 요구사항

-   **Python 3.8+**
-   **Google Gemini API Key** (환경 변수 `GEMINI_API_KEY`로 설정)

### 2. 설치

```bash
# 가상 환경 활성화 (권장)
source venv/bin/activate 

# 필수 라이브러리 설치
pip install -r requirements.txt
```
---
