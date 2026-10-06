from contextlib import asynccontextmanager
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from dotenv import load_dotenv

import time
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
    
    #loading model onto ExLlamaV2
    print(f"Loading model from : {MODEL_DIR}...")
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
    
    state["generator"] = generator
    state["tokenizer"] = tokenizer
    print("Model loaded on VRAM succesfully.")
    yield #loading completed
    
    #free() state (in wich the model was loaded in) when apllication shuts down
    state.clear()


app = FastAPI(title="Local LLM Engine", lifespan=lifespan)

#local frontend CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        
        # Exits when STOP token
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok", "gpu_ready": "generator" in state}