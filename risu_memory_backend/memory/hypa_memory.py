import os
import google.generativeai as genai
import numpy as np
from typing import List, Dict, TypedDict, Set, Optional

# 경로 수정
from ..tokenizer import Tokenizer, count_tokens


# --- Data Structures ---

class HypaV3Settings(TypedDict):
    summarization_model: str
    embedding_model: str
    summarization_prompt: str
    memory_tokens_ratio: float
    max_chats_per_summary: int
    recent_memory_ratio: float
    similar_memory_ratio: float


class Summary(TypedDict):
    text: str
    chat_memos: Set[str]
    is_important: bool
    embedding: Optional[List[float]]
    # 이 필드는 계산 중에만 사용되므로 Optional로 유지합니다.
    similarity_score: Optional[float]


class HypaV3Data(TypedDict):
    summaries: List[Summary]


class OpenAIChat(TypedDict):
    role: str
    content: str
    memo: Optional[str]


class Chat(TypedDict):
    hypaV3Data: Optional[HypaV3Data]


# --- Gemini API Configuration ---
if api_key := os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY environment variable not set.")

tokenizer = Tokenizer()


# --- Helper Functions ---

def get_embedding(text: str, model: str = "text-embedding-004") -> List[float]:
    """Generates embedding for a given text using Gemini API."""
    text = text.replace("\n", " ")
    result = genai.embed_content(model=model, content=text)
    return result['embedding']


def similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    return np.dot(v1, v2)


async def summarize_for_hypa(text_to_summarize: str, settings: HypaV3Settings) -> str:
    """Summarizes text specifically for HypaMemory, using its settings."""
    prompt = settings['summarization_prompt']
    if not prompt:
        prompt = "[Summarize the ongoing role story, It must also remove redundancy and unnecessary text and content from the output.]"

    full_prompt = f"{text_to_summarize}\n\n{prompt}\n\nOutput:"

    try:
        model_name = settings['summarization_model']
        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async(full_prompt)
        result = response.text.strip()
        return result if result else text_to_summarize
    except Exception as e:
        print(f"Error during Hypa summarization: {e}")
        return text_to_summarize  # Fallback


# --- Core HypaMemoryV3 Logic ---

async def hypa_memory_v3(
        chats: List[OpenAIChat],
        current_tokens: int,
        max_context_tokens: int,
        room: Chat,
        settings: HypaV3Settings,
) -> dict:
    log_prefix = "[HypaV3]"
    memory_prompt_tag = "Past Events Summary"
    min_chats_for_similarity = 3

    data: HypaV3Data = room.get('hypaV3Data') or {'summaries': []}

    # --- 1. Summarization Phase ---
    start_idx = 0
    if data['summaries']:
        last_summary = data['summaries'][-1]
        try:
            last_chat_index = next(
                i for i, chat in reversed(list(enumerate(chats))) if chat.get('memo') in last_summary['chat_memos'])
            start_idx = last_chat_index + 1
            summarized_chats = chats[:start_idx]
            for chat in summarized_chats:
                current_tokens -= tokenizer.count_chat_tokens(chat)
        except StopIteration:
            pass

    print(
        f"{log_prefix} Starting with {len(data['summaries'])} summaries. Current tokens: {current_tokens}. Analyzing from chat index {start_idx}.")

    if current_tokens > max_context_tokens:
        print(f"{log_prefix} Context limit exceeded. Starting summarization process.")
        to_summarize_batch: List[OpenAIChat] = []
        tokens_to_be_removed = 0

        for i in range(start_idx, len(chats) - min_chats_for_similarity):
            chat = chats[i]
            if len(to_summarize_batch) < settings['max_chats_per_summary']:
                if chat.get('memo') == 'NewChat' or not chat.get('content', '').strip():
                    continue
                to_summarize_batch.append(chat)
                tokens_to_be_removed += tokenizer.count_chat_tokens(chat)
            else:
                break

        if to_summarize_batch:
            stringlized_chat = "\n".join([f"{c['role']}: {c['content']}" for c in to_summarize_batch])

            summary_text = await summarize_for_hypa(stringlized_chat, settings)
            summary_embedding = get_embedding(summary_text, model=settings['embedding_model'])

            new_summary = Summary(
                text=summary_text,
                chat_memos={chat.get('memo') for chat in to_summarize_batch if chat.get('memo')},
                is_important=False,
                embedding=summary_embedding
            )
            data['summaries'].append(new_summary)

            current_tokens -= tokens_to_be_removed
            start_idx += len(to_summarize_batch)
            print(
                f"{log_prefix} Created new summary. Total summaries: {len(data['summaries'])}. Current tokens: {current_tokens}.")

    # --- 2. Memory Selection Phase ---
    memory_content = ""
    if data['summaries']:
        available_memory_tokens = max_context_tokens * settings['memory_tokens_ratio']
        selected_summaries: List[Summary] = []

        recent_chats_for_query = [c for c in chats[-min_chats_for_similarity:] if c.get('content', '').strip()]
        if recent_chats_for_query:
            query_text = "\n".join([c['content'] for c in recent_chats_for_query])
            query_embedding = get_embedding(query_text, model=settings['embedding_model'])

            for summary in data['summaries']:
                if summary.get('embedding'):
                    summary['similarity_score'] = similarity(query_embedding, summary['embedding'])
                else:
                    summary['similarity_score'] = 0
        else:
            for summary in data['summaries']:
                summary['similarity_score'] = 0

        similar_candidates = sorted(data['summaries'], key=lambda s: s.get('similarity_score', 0), reverse=True)
        recent_candidates = sorted(data['summaries'], key=lambda s: data['summaries'].index(s), reverse=True)

        consumed_tokens = 0

        important_summaries = [s for s in data['summaries'] if s['is_important']]
        for summary in important_summaries:
            summary_tokens = count_tokens(summary['text'])
            if consumed_tokens + summary_tokens <= available_memory_tokens and summary not in selected_summaries:
                selected_summaries.append(summary)
                consumed_tokens += summary_tokens

        similar_memory_tokens_limit = available_memory_tokens * settings['similar_memory_ratio']
        for summary in similar_candidates:
            summary_tokens = count_tokens(summary['text'])
            if consumed_tokens + summary_tokens <= available_memory_tokens and summary not in selected_summaries:
                if (consumed_tokens + summary_tokens) <= similar_memory_tokens_limit:
                    selected_summaries.append(summary)
                    consumed_tokens += summary_tokens

        recent_memory_tokens_limit = available_memory_tokens * settings['recent_memory_ratio']
        for summary in recent_candidates:
            summary_tokens = count_tokens(summary['text'])
            if consumed_tokens + summary_tokens <= available_memory_tokens and summary not in selected_summaries:
                if (consumed_tokens + summary_tokens) <= recent_memory_tokens_limit:
                    selected_summaries.append(summary)
                    consumed_tokens += summary_tokens

        selected_summaries.sort(key=lambda s: data['summaries'].index(s))
        memory_content = "\n\n".join([s['text'] for s in selected_summaries])

    # --- 3. Final Assembly Phase ---
    final_memory_prompt = f"<{memory_prompt_tag}>\n{memory_content}\n</{memory_prompt_tag}>" if memory_content else ""
    final_memory_tokens = count_tokens(final_memory_prompt)

    final_chats = chats[start_idx:]
    final_tokens = sum(tokenizer.count_chat_tokens(c) for c in final_chats) + final_memory_tokens

    while final_tokens > max_context_tokens and len(final_chats) > 1:
        removed_chat = final_chats.pop(0)
        final_tokens -= tokenizer.count_chat_tokens(removed_chat)

    if final_memory_prompt:
        final_chats.insert(0, {"role": "system", "content": final_memory_prompt, "memo": "hypaMemory"})

    print(f"{log_prefix} Final context ready. Tokens: {final_tokens}. Chats: {len(final_chats)}.")

    return {
        "current_tokens": final_tokens,
        "chats": final_chats,
        "memory_data": data,
        "error": None
    }