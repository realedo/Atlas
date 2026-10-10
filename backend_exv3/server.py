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




#
import time
import json
import re
#
from model_loader import load_ai_model, clear_ai_model, state
from config import default_max_tok_out, default_temp, WEB_TOGGLE, MAX_CONTEXT, min_free_tokens_for_inf, max_url_scrape_len
from prompt_formatter import Message, build_chatML_prompt
from web_search import web_search, search_url
from query_optimizer import gen_optimized_query_to_search



url_pattern = re.compile(r'(https?://[^\s]+)')

inference_lock = threading.Lock()



@asynccontextmanager
async def lifespan(app: FastAPI):
    success = load_ai_model()
    if not success:
        print("[Error] Model did not load correctly")
        
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

def inference_chunk(content = None, finish_reason=None):
    return{
               
        "choices": [
            {
                "index": 0,
                "delta": {} if content is None else {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    }



#inference endopoint (only local for now see cors settings). post only
@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatCompletionRequest): #req structure declared before in the state dict
    
    generator = state.get("generator")
    tokenizer = state.get("tokenizer")
    
    if (generator is None) or (tokenizer is None):
        raise HTTPException(status_code=503, detail="Model not ready/loaded unsuccesfully.")

    #get the last user's message
    last_user_message = next((msg.content for msg in reversed(req.messages) if msg.role == "user"), "")
           
    modified_messages = list(req.messages)
    
    web_context=""
    if WEB_TOGGLE and last_user_message.strip():
        print("Starting web search...\n")
        web_start_time = time.time()
        
        #checks if theres URLs in the prompt:
        try:
            urls_present = url_pattern.findall(last_user_message)
            if urls_present:
                url_to_search = urls_present[0]
                print(f"Url ({url_to_search}) found in prompt, proceding to scrape from said url...\n")
                web_context = search_url(url_to_search)
                    
            else:
                print(f"No direct URLs found in prompt. generating a optimized search query...\n")
                
                with inference_lock: #cant inference both gen_optimized_query_to_search() and event_stream()
                    optimized_query = gen_optimized_query_to_search(
                        generator,
                        tokenizer,
                        last_user_message,
                    )

                print(f"Searching for: {optimized_query}")
                web_context = web_search(optimized_query)

        except Exception as e:
            print(f"[Warning] Failed web search: {str(e)}")
            web_context = ""

        
        print(f"Searched for {time.time()- web_start_time} seconds...\n")
    
    if web_context and not web_context.startswith(("Web search initialization failed:", "Web search is disabled")):
        #add web search
        modified_messages.insert(0,
                                 Message(
                                     role="system",
                                     content=("You have access to the following web search results. Use them when they are relevant to the user's question."
                                              "Prioritize the supplied sources for current or time-sensitive facts, but do not assume every result is accurate."
                                              "Do not invent facts or claim that a source supports something it does not. If the results are insufficient, say so."
                                              "Always provide extensive, highly detailed, and comprehensive answers. Never give brief or single-sentence responses."
                                              "Cite relevant sources if you absolutly deem necessary to, using their supplied URLs.\n\n" f"{web_context}")
                                 )
                                )

    formatted_prompt = build_chatML_prompt(modified_messages) #format before putting it into the model

    def event_stream():
        
        started = False
        job = None
        
        try:
            
            #encode prompt
            with inference_lock:
                input_ids = tokenizer.encode(
                    formatted_prompt,
                    encode_special_tokens=True,
                )
            
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
                yield "data: [DONE]\n\n"
                return
            
            requested_tokens = (
                    req.max_tokens
                    if req.max_tokens is not None
                    else default_max_tok_out
                )
            max_new_tokens = min(requested_tokens, available_tokens - 16)
            
            temperature = (
                    req.temperature
                    if req.temperature is not None
                    else default_temp
                )

            
            stop_conditions = ["<|im_end|>"]
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                    stop_conditions.append(int(eos_token_id))

            sampler = ComboSampler(
                    temperature=float(temperature),
                )
            
            job = Job(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    sampler=sampler,
                    stop_conditions=stop_conditions,
                    decode_special_tokens=False,
                )

            generator.enqueue(job)

            
            print(
                    f"\nStarting EXL3 inference "
                    f"(prompt: {prompt_len}/{MAX_CONTEXT} tokens, "
                    f"max output: {max_new_tokens}, "
                    f"temperature: {temperature})"
                )

            start_time = time.time()
            generated_tokens = 0
            finish_reason = "stop"
            
            thinking = False
            buffer = ""

            while generator.num_remaining_jobs():
                    results = generator.iterate()

                    for result in results:
                        if result.get("stage") == "error":
                            raise result["error"]

                        if result.get("stage") != "streaming":
                            continue

                        chunk = result.get("text", "")
                        if chunk:
                            #generated_tokens += int(
                            #    result.get("new_tokens", 0)
                            #) if result.get("eos") else 0
                            generated_tokens +=1
                            print(f"\rGenerating tokens... ({generated_tokens} / {max_new_tokens})", end="", flush=True)
                            
                            #I dont want CoT to be printed --> buffer
                            buffer += chunk
                            
                            while buffer:
                                #if not thiking, check if <think> nearby
                                if not thinking:
                                    if "<think>" in buffer:
                                        
                                        before, buffer = buffer.split("<think>", 1)
                                        if before:
                                            yield sse(inference_chunk(content=before))
                                        
                                        yield sse(inference_chunk(content="*Making CoT...*\n\n"))
                                           
                                        thinking = True
                                    
                                    else:
                                        
                                        safe_to_yield = buffer
                                        
                                        #if it ends with <think> 
                                        for i in range(1, 7):
                                            if buffer.endswith("<think>"[:i]):
                                                safe_to_yield = buffer[:-i]
                                                buffer = buffer[-i:]
                                                break
                                        else:
                                            buffer = ""
                                            
                                        if safe_to_yield:
                                            yield sse(inference_chunk(content=safe_to_yield))
                                        break
                                else:
                                    #in CoT mode
                                    if "</think>" in buffer:
                                        
                                        _, buffer = buffer.split("</think>", 1)
                                        thinking = False
                                    else:
                                        
                                        for i in range(1, 8):
                                            if buffer.endswith("</think>"[:i]):
                                                buffer = buffer[-i:]
                                                break
                                        else:
                                            buffer = ""
                                        break
                           
                        if result.get("eos"):
                            reason = result.get("eos_reason", "stop")
                            if reason == "max_new_tokens":
                                finish_reason = "length"
                            else:
                                finish_reason = "stop"
                            break

            if buffer and not thinking:
                yield sse(inference_chunk(content=buffer))

                
            elapsed_time = time.time() - start_time
            total_generated = getattr(job, "new_tokens", generated_tokens)
            tps = total_generated / elapsed_time

            print(
                    f"\nGeneration finished: {total_generated} tokens, "
                    f"{tps:.2f} tokens/s, {elapsed_time:.2f}s"
                )

            yield sse(inference_chunk(finish_reason=finish_reason))

        
        #bug fix from v2, stop inference if clients disconnects
        except GeneratorExit:
            if started:
                try:
                    job = locals().get("job")
                    if job is not None and generator.num_remaining_jobs():
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
        #general exeption
        except Exception as exc:
            print(f"[ERROR] Inference failed: {exc}")
            yield sse({
                "error": {
                    "message": f"Inference failed: {str(exc)}",
                    "type": "server_error",
                }
            })

        finally:
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