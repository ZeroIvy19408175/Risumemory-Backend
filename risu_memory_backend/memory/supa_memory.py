import os
import google.generativeai as genai
from typing import List, Dict, TypedDict, Optional

# 경로 수정
from ..tokenizer import count_tokens, count_chat_tokens


# --- Data Structures ---
class OpenAIChat(TypedDict):
    role: str
    content: str
    memo: Optional[str]


class Chat(TypedDict):
    supaMemoryData: Optional[str]


class Character(TypedDict):
    name: str


# --- Gemini API Configuration ---
# 실행 전 터미널에서 `export GEMINI_API_KEY='YOUR_API_KEY'` 를 설정해야 합니다.
if api_key := os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY environment variable not set.")


# --- Core Functions ---
async def summarize(text_to_summarize: str, supa_model_type: str = 'gemini-1.5-flash-latest',
                    supa_memory_prompt: str = "") -> str:
    """
    Summarizes the given text using the Gemini API.
    """
    if not supa_memory_prompt:
        supa_memory_prompt = "[Summarize the ongoing role story, It must also remove redundancy and unnecessary text and content from the output to reduce tokens for gpt3 and other sublanguage models]"

    prompt_body = f"{text_to_summarize}\n\n{supa_memory_prompt}\n\nOutput:"

    try:
        # Gemini 모델 호출로 변경
        model = genai.GenerativeModel(supa_model_type)
        response = await model.generate_content_async(prompt_body)

        result = response.text.strip()
        if not result:
            raise ValueError("Summarization returned an empty result.")
        return result
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text_to_summarize


async def supa_memory(
        chats: List[OpenAIChat],
        current_tokens: int,
        max_context_tokens: int,
        room: Chat,
        char: Character,
) -> dict:
    """
    Manages long-term memory by summarizing the conversation.
    """
    if current_tokens <= max_context_tokens:
        return {
            "current_tokens": current_tokens,
            "chats": chats,
            "error": None
        }

    print(
        f"Context limit exceeded. Current tokens: {current_tokens}. Max tokens: {max_context_tokens}. Starting summarization.")

    supa_memory_summary = ''
    last_id = ''

    if room.get('supaMemoryData') and len(room['supaMemoryData']) > 4:
        supa_memory_summary = room['supaMemoryData']
        current_tokens += count_tokens(supa_memory_summary)

    while current_tokens > max_context_tokens:
        chunk_size = 0
        stringlized_chat = ''
        splice_len = 0
        max_chunk_tokens = max_context_tokens / 3

        for i, chat_message in enumerate(chats):
            if chat_message['role'] == 'system':
                continue

            message_tokens = count_chat_tokens(chat_message)
            if chunk_size + message_tokens > max_chunk_tokens and stringlized_chat:
                last_id = chat_message.get('memo', '')
                break

            stringlized_chat += f"{char['name'] if chat_message['role'] == 'assistant' else 'user'}: {chat_message['content']}\n\n"
            splice_len = i + 1
            current_tokens -= message_tokens
            chunk_size += message_tokens

        if not stringlized_chat:
            return {
                "current_tokens": current_tokens,
                "chats": chats,
                "error": "Not enough tokens to summarize or failed to create a summarization chunk."
            }

        chats = chats[splice_len:]

        new_summary_part = await summarize(stringlized_chat)
        new_summary_tokens = count_tokens(new_summary_part)

        supa_memory_summary = f"{supa_memory_summary}\n\n{new_summary_part}".strip()
        current_tokens += new_summary_tokens

    chats.insert(0, {
        "role": "system",
        "content": supa_memory_summary,
        "memo": "supaMemory"
    })

    return {
        "current_tokens": current_tokens,
        "chats": chats,
        "memory": supa_memory_summary,
        "last_id": last_id,
        "error": None
    }