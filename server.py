from contextlib import asynccontextmanager
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from dotenv import load_dotenv

import time
import torch
import os
import json


#----env varaibles/secrets
load_dotenv()
MODEL_DIR = os.environ.get("dir_model")
default_temp = 0.7
default_max_tok_out = 1024
#----
#---- model settings ---
MAX_CONTEXT = 4096 #trial and error for my gpu
#---

state = {} # dynamic global dict. (persist on verious HTTPS req.)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    if not MODEL_DIR or not os.path.exists(MODEL_DIR):
        print(f"model directory invalid or missing : {MODEL_DIR}")
        yield #fails health check
        return
    #loading model onto ExLlamaV2
    print(f"Loading model from : {MODEL_DIR}...")
    
    try:
        
        config = ExLlamaV2Config(MODEL_DIR) #loads model's config
        config.max_seq_len = MAX_CONTEXT #sets max context
        
        #malloc for model
        model = ExLlamaV2(config)
        #lazy cache (sequential, to avoid fragmentaion errors)
        cache = ExLlamaV2Cache(model, max_seq_len=MAX_CONTEXT, lazy=True)
        #loads weights on VRAM
        model.load_autosplit(cache) 
        
        tokenizer = ExLlamaV2Tokenizer(MODEL_DIR)
        generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
        
        #makes dict.
        state["generator"] = generator
        state["tokenizer"] = tokenizer
        print("Model loaded on VRAM succesfully.")
    
    except torch.cuda.OutOfMemoryError:
        print(f"VRAM OoM during loading. Try lowering {MAX_CONTEXT}")
        #free corrupted malloc(s)
        torch.cuda.empty_cache() 
    
    except Exception as e:
        print(f"Failed to load model. -> {str(e)}")
    
    yield #loading completed (succesfully)
    
    #free() state (in wich the model was loaded in) when apllication shuts down, also free empty cache
    state.clear()
    torch.cuda.empty_cache()


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
class Message(BaseModel):  #role + message
    role: Literal["system", "user", "assistant"]
    #the actual message
    content: str

class ChatCompletionRequest(BaseModel): #completions
    messages: List[Message]
    
    #settings that can be forwarded in the request. other wise default settings.
    max_tokens: Optional[int] = default_max_tok_out
    temperature: Optional[float] = default_temp
    
    stream: Optional[bool] = True


#fromatting JSON input into allowed format for the model (chatML)-->(https://web.archive.org/web/20230303120844/https://github.com/openai/openai-python/blob/main/chatml.md) (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)
def build_chatML_prompt(messages: List[Message]) -> str:
   
    prompt = "" #blank promptm, append as you go
    for msg in messages:
       
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        
    prompt += "<|im_start|>assistant\n"
    return prompt



#inference endopoint (only local for now see cors settings). post only
@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatCompletionRequest): #req structure devlared before
    generator = state.get("generator")
    if not generator:
        raise HTTPException(status_code=503, detail="Model not ready/loaded unsuccesfully.")

    formatted_prompt = build_chatML_prompt(req.messages) #format before putting it into the model

    def event_stream():
        
        try:
            
            #use ExLlamaV2 ExLlamaV2DynamicGenerator.generate_text syncronously
            for chunk in generator.generate_text(
                prompt=formatted_prompt,
                max_new_tokens=req.max_tokens, #hard limit on max tokens (comes from JSON injest or default settings.)
                stream=True #returns to pyhton after each token (token streaming, it gets formattesd to payload and posted)
            ):
                #formats payload for post (def OpenAI format)
                payload = {
                    "choices": [{
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(payload)}\n\n" #returns payload currently generated
        
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