import torch
import threading
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from exllamav3.generator import Job
from exllamav3.generator.sampler import ComboSampler

import time
import json
import re

from model_loader import load_ai_model, clear_ai_model, state
from config import default_max_tok_out, default_temp, MAX_CONTEXT, min_free_tokens_for_inf, max_url_scrape_len, min_tokens_think
from prompt_formatter import Message, build_chatML_prompt
from masker import ToolFilter, CoTFilter
from sliding_window import prune_messages_to_fit
from search_appender import add_web_context

url_pattern = re.compile(r'(https?://[^\s]+)')

inference_lock = threading.Lock()

#print() colors
INFO = "\033[92m[Info]\033[0m"
WARN = "\033[93m[Warning]\033[0m"
ERR = "\033[91m[Error]\033[0m"





@asynccontextmanager
async def lifespan(app: FastAPI):
    success = load_ai_model()
    if not success:
        print(f"{ERR} Model did not load correctly")
        
    try:
        yield  
    finally:
        clear_ai_model()


app = FastAPI(title="Local LLM Engine", lifespan=lifespan)

#local frontend CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400, #bad req (formatting)
        content={
            "error": {
                "message": "Invalid JSON payload or missing required fields.",
                "type": "invalid_request_error",
                "details": exc.errors()
            }
        },
    )


#---
#JSON ingest structure / defualt OpenAI structure
#---

class ChatCompletionRequest(BaseModel): #completions
    messages: List[Message]
    
    #settings that can be forwarded in the request. other wise default settings.
    max_tokens: Optional[int] = Field(default=default_max_tok_out, ge=1)
    temperature: Optional[float] = Field(default=default_temp, ge=0.0, le=2.0) #industry standard constarints (openAi ones)
    stream: Optional[bool] = True

#server sent event
def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"





#inference endopoint (only local for now see cors settings). post only
@app.post("/v1/chat/completions")
async def chat_completions(raw_req: Request):

    request_id = f"chatcmpl-{int(time.time()*1000)}"
    created_time = int(time.time())
        
    def inference_chunk(content=None, tool_call=None, finish_reason=None, role=None):
        delta = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if tool_call is not None:
            delta["tool_calls"] = [tool_call]
            
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "exllamav3-local",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ]
        }
    
    
    generator = state.get("generator")
    tokenizer = state.get("tokenizer")
    
    if (generator is None) or (tokenizer is None):
        raise HTTPException(status_code=503, detail="Model not ready/loaded unsuccessfully.")

    data = await raw_req.json()
    
    safe_messages = []
    for m in data.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")
        tool_call_id = m.get("tool_call_id")
        
        if isinstance(content, list):
            parsed_content = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parsed_content.append(item["text"])
                else:
                    parsed_content.append(json.dumps(item, ensure_ascii=False))
            content = " ".join(parsed_content)
        elif not content:
            content = ""
        else:
            content = str(content)
            
        safe_messages.append(Message(
            role=role, 
            content=content, 
            tool_calls=tool_calls, 
            tool_call_id=tool_call_id
        ))
        
    
    req_max_tokens = data.get("max_tokens")
    if req_max_tokens is None:
        req_max_tokens = default_max_tok_out
    else:
        req_max_tokens = max(int(req_max_tokens), min_tokens_think)
        
    req_max_tokens = min(req_max_tokens, default_max_tok_out)
    
    
    
    req_temperature = data.get("temperature", default_temp)
    
    #get the last user's message
    last_user_message = next((msg.content for msg in reversed(safe_messages) if msg.role == "user"), "")
    
    add_web_context(last_user_message, inference_lock, generator, tokenizer, safe_messages)
    
    
    tools = data.get("tools", [])
    if tools:
        tool_prompt = (
            "You have access to the following tools:\n\n"
            f"<tools>\n{json.dumps(tools, ensure_ascii=False, indent=2)}\n</tools>\n\n"
            "To use a tool, you MUST respond using the following XML format:\n"
            "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg_name\": \"value\"}}\n</tool_call>\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. You may ONLY use the tools explicitly listed above. Do not hallucinate or invent unlisted tools.\n"
            "2. Do not output any conversational text before or after the tool call XML block."
        )
        safe_messages.insert(0, Message(role="system", content=tool_prompt))
        
        
        
    safe_messages = prune_messages_to_fit(
            safe_messages, 
            tokenizer, 
            max_allowed_tokens=(MAX_CONTEXT - min_free_tokens_for_inf)
        )
    
    
    
    formatted_prompt = build_chatML_prompt(safe_messages) 
    input_ids = tokenizer.encode(formatted_prompt, encode_special_tokens=True)
    prompt_len = input_ids.shape[-1]
    available_tokens = MAX_CONTEXT - prompt_len

    if available_tokens < min_free_tokens_for_inf:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Context too large: prompt uses {prompt_len}/{MAX_CONTEXT} tokens.",
                    "type": "invalid_request_error"
                }
            }
        )
    def event_stream():
        
        with inference_lock:
            started = False
            job = None
            
            try:
                
                ##---- SETTINGS AND CONTEX WINDOW HANDLING ----#
                
                # !! check if the prompt is not out of bound of max context lenght allowed !! #
                prompt_len = input_ids.shape[-1]
                #check how much space is left 
                available_tokens = MAX_CONTEXT - prompt_len
                
                if available_tokens < min_free_tokens_for_inf:
                    message = (
                            f"Context too large: prompt uses {prompt_len} tokens "
                            f"out of {MAX_CONTEXT}. Available generation space: "
                            f"{available_tokens}. Reduce the conversation or web "
                            f"context, increase MAX_CONTEXT if memory permits, "
                            f"or lower min_free_tokens_for_inf. "
                            f"Configured scrape limit: {max_url_scrape_len}."
                        )
                    yield sse({"error": {"message": message}})
                    return
                
                requested_tokens = req_max_tokens if req_max_tokens is not None else default_max_tok_out
                max_new_tokens = min(requested_tokens, available_tokens - 16)
                
                stop_conditions = ["<|im_end|>"]
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                        stop_conditions.append(int(eos_token_id))

                sampler = ComboSampler(
                        temperature=float(req_temperature),
                    )
                
                get_cache_stats = getattr(generator, "get_cache_stats", None)
                cache_before = (
                    get_cache_stats()
                    if callable(get_cache_stats)
                    else None
                )
                                
                
                
                job = Job(
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        sampler=sampler,
                        stop_conditions=stop_conditions,
                        decode_special_tokens=False,
                    )

                generator.enqueue(job)
                started = True
                
                yield sse(inference_chunk(role="assistant"))
                
                print(
                        f"\n{INFO} Starting EXL3 inference "
                        f"(prompt: {prompt_len}/{MAX_CONTEXT} tokens, "
                        f"max output: {max_new_tokens}, "
                        f"temperature: {req_temperature})\nStarting prefill...\n"
                    )

                start_time = time.time()
                generation_start_time = start_time
                prefill_done = False
                time_to_prefill = 0.0
                
                generated_tokens = 0
                finish_reason = "stop"
                was_thinking = False
                
                cot_masker = CoTFilter()
                cot_tokens=0
                tool_masker = ToolFilter()
                did_tool_call = False
                full_model_response = ""

                while generator.num_remaining_jobs():
                        results = generator.iterate()

                        for result in results:
                            
                            if result.get("job") is not job:
                                continue

                            if result.get("stage") == "error":
                                raise result["error"]

                            if result.get("stage") != "streaming":
                                continue
                            if result.get("job") is job and result.get("eos"):
                                print(
                                    "\n[KV cache] "
                                    f"cached_tokens={result.get('cached_tokens', 0)}, "
                                    f"cached_pages={result.get('cached_pages', 0)}, "
                                    f"prefill_time={result.get('time_prefill', 'n/a')}s"
                                )

                            chunk = result.get("text", "")
                            
                            if not prefill_done and (chunk or result.get("eos")):
                                time_to_prefill = time.time() - start_time
                                read_tps = prompt_len / time_to_prefill if time_to_prefill > 0 else 0
                                print(f"{INFO} Done prefilling prompt in {time_to_prefill:.2f} seconds : (read {read_tps:.2f} tok/sec)\n")
                                generation_start_time = time.time()
                                prefill_done = True
                            
                            
                            
                            if chunk:
                                generated_tokens +=1
                                if (generated_tokens % 100) == 0:  # Only print every 150 tokens
                                    print(f"\r{INFO} Generating tokens... ({generated_tokens} / {max_new_tokens})", end="")
                                was_thinking = cot_masker.in_think
                                filtered_output = cot_masker.process_chunk(chunk)
                                
                                #if CoT just started
                                if not was_thinking and cot_masker.in_think:
                                    #yield sse(inference_chunk(content="\n*CoT phase... 💭*\n\n"))
                                    pass

                                #If acttivly in Cot
                                if cot_masker.in_think or (was_thinking and not cot_masker.in_think):
                                    cot_tokens += 1

                                #Cot finished
                                if was_thinking and not cot_masker.in_think:
                                    print(f"\n{INFO} CoT phase done. used : {cot_tokens} to think\n")

                                #send output
                                if filtered_output:
                                    text_out, tool_out = tool_masker.process_chunk(filtered_output)
                                    
                                    if text_out:
                                        full_model_response += text_out
                                        yield sse(inference_chunk(content=text_out))
                                        
                                    if tool_out:
                                        did_tool_call = True
                                        print(f"\n{WARN} Executing tool: {tool_out['function']['name']}\n")
                                        yield sse(inference_chunk(tool_call=tool_out))
                            
                            if result.get("eos"):
                                reason = result.get("eos_reason", "stop")
                                if reason == "max_new_tokens":
                                    finish_reason = "length"
                                else:
                                    
                                    finish_reason = "tool_calls" if did_tool_call else "stop"
                                break
                
                            
                if cache_before is not None:
                    cache_after = get_cache_stats()

                    reused_pages = (
                        cache_after["alloc_cached_pages"]
                        - cache_before["alloc_cached_pages"]
                    )
                    allocated_pages = (
                        cache_after["alloc_pages"]
                        - cache_before["alloc_pages"]
                    )
                    page_size = cache_after["page_size"]

                print(
                    f"{INFO} Prefix cache: reused {reused_pages} pages "
                    f"(~{reused_pages * page_size} tokens); "
                    f"allocated {allocated_pages} pages"
                )
                            
                            
                            
                is_length_truncated = (finish_reason == "length")
                complete_output = cot_masker.flush(force=is_length_truncated)
                
                if complete_output:
                    if is_length_truncated and cot_masker.in_think:
                        complete_output = f"\n\n[System: Generation truncated by max_tokens limit while thinking.]\n"
                    full_model_response += complete_output
                    yield sse(inference_chunk(content=complete_output))
                    
                elapsed_time = time.time() - start_time
                total_generated = getattr(job, "new_tokens", generated_tokens)
                tps = total_generated / elapsed_time if elapsed_time > 0 else 0

                print(
                        f"\n{INFO} Generation finished: {total_generated} tokens, "
                        f"{tps:.2f} tokens/s, {elapsed_time:.2f}s"
                    )
                
                yield sse(inference_chunk(finish_reason=finish_reason))
                
                yield "data: [DONE]\n\n"

            
            #bug fix from v2, stop inference if clients disconnects
            except GeneratorExit:
                if started and job is not None:
                    try:
                        if generator.num_remaining_jobs():
                            generator.cancel(job)
                    except Exception:
                        pass
                raise
            #OoM
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                yield sse({
                    "error": {
                        "message": (
                            "CUDA out of memory. Reduce MAX_CONTEXT, max_tokens, "
                            "or other VRAM usage."
                        ),
                        "type": "server_error",
                    }
                })
                yield "data: [DONE]\n\n"
                
            #general exeption
            except Exception as exc:
                print(f"{ERR} Inference failed: {exc}")
                yield sse({
                    "error": {
                        "message": f"Inference failed: {str(exc)}",
                        "type": "server_error",
                    }
                })
                yield "data: [DONE]\n\n"


    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/health")
def health():
    ready = (
        state.get("generator") is not None
        and state.get("tokenizer") is not None
    )
    return {
        "status": "ok" if ready else "not_ready",
        "gpu_ready": ready,
    }