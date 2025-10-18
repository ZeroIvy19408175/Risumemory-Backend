# 파일명: api.py (최상위 디렉토리)

from fastapi import FastAPI
import os
import google.generativeai as genai
from risu_memory_backend.memory.hypa_memory import hypa_memory_v3 # 메모리 모듈 임포트
from risu_memory_backend.memory.supa_memory import supa_memory

# --- 1. FastAPI App 객체 생성 ---
app = FastAPI(title="RisuMemoryBackend API", version="0.1.0")

# --- 2. API 설정 (main.py 없이 여기서 처리) ---
if api_key := os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
    print("API Key loaded and Gemini configured.")
else:
    print("Warning: GEMINI_API_KEY environment variable not set. API will run, but memory functions will fail.")

# --- 3. 기본 라우트 (상태 확인용) ---
@app.get("/")
def read_root():
    return {"status": "ok", "service": "RisuMemoryBackend", "version": "0.1.0"}

# --- 4. 메모리 관련 라우트 (여기에 HypaV3, SupaMemory 호출 로직 추가) ---
# 예시: 메모리 호출 테스트 라우트
# @app.post("/memory/process")
# async def process_memory_context(data: MemoryRequestSchema):
#     # HypaV3 또는 SupaMemory 함수를 호출하는 로직이 여기에 들어갑니다.
#     # ...
#     pass

if __name__ == "__main__":
    import uvicorn
    # 로컬 테스트용
    uvicorn.run(app, host="0.0.0.0", port=8000)