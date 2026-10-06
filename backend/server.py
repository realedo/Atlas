from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from exllamav2.generator import ExLlamaV2Sampler
#
from ddgs import DDGS
import time
import torch
import json
#
from model_loader import load_ai_model, clear_ai_model, state
from config import default_max_tok_out, default_temp, WEB_TOGGLE
from prompt_formatter import Message, build_chatML_prompt
from web_search import web_search

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    success = load_ai_model()
    if not success:
        yield
        return
    yield
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
    max_tokens: Optional[int] = default_max_tok_out
    temperature: Optional[float] = default_temp
    
    stream: Optional[bool] = True



#inference endopoint (only local for now see cors settings). post only
@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatCompletionRequest): #req structure declared before in the state dict
    generator = state.get("generator")
    tokenizer = state.get("tokenizer")
    
    if not generator or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not ready/loaded unsuccesfully.")

    #get the last user's message
    last_user_message = next((msg.content for msg in reversed(req.messages) if msg.role == "user"), "")
           
    modified_messages = list(req.messages)
    
    web_context=""
    if WEB_TOGGLE and last_user_message.strip():
        web_start_time = time.time()
        print(f"Searching for {last_user_message}\n")
        web_context = web_search(last_user_message)
        print(f"Searched for {time.time()- web_start_time} seconds\n")
    
    if web_context and not web_context.startswith(("Web search initialization failed:", "Web search is disabled")):
        #add web search
        modified_messages.insert(0,
                                 Message(
                                     role="system",
                                     content=("You have access to the following web search results. Use them when they are relevant to the user's question."
                                              "Prioritize the supplied sources for current or time-sensitive facts, but do not assume every result is accurate."
                                              "Do not invent facts or claim that a source supports something it does not. If the results are insufficient, say so."
                                              "Cite relevant sources if you deem necessary using their supplied URLs.\n\n" f"{web_context}")
                                 )
                                )

    formatted_prompt = build_chatML_prompt(modified_messages) #format before putting it into the model

    def event_stream():
        
        try:
            
            settings = ExLlamaV2Sampler.Settings()
            settings.temperature = req.temperature
            
            input_ids = tokenizer.encode(formatted_prompt)
            
            generator.begin_stream(input_ids, settings)
            
            tokens_generated = 0
            start_time = time.time()
            print("\nStarted Inference\n", end="", flush=True)
            
            while True:
                
                chunk, eos, _ = generator.stream()
                
                if chunk:
                    payload = {
                        "choices": [{
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                tokens_generated +=1
                print(f"\rTokens generated: {tokens_generated}", end="", flush=True)
                
                if eos or tokens_generated>=req.max_tokens:
                    break
                
            elapsed_time = time.time() - start_time
            tps = tokens_generated/elapsed_time if elapsed_time>= 0 else 0
            print(f"\ngenerated {tokens_generated} tokens. tps = {tps}")                       
        
        
        #OoM
        except torch.cuda.OutOfMemoryError:
            #free memory
            torch.cuda.empty_cache()
            error_payload = {
                "error": "VRAM OoM. Conversation context exeeded avalable memory."
            }
            yield f"data: {json.dumps(error_payload)}\n\n"   
        #general error  
        except Exception as e:
            error_payload = {
                "error": f"Internal inference error -> {str(e)}"
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
        #close the stream
        finally:
            # Exits when STOP token
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok", "gpu_ready": "generator" in state} #fails if generator is yielded as empty (expetion as e)