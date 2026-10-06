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