import torch
import os
from exllamav2 import ExLlamaV2, ExLlamaV2Cache_Q8, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator
from config import MODEL_DIR, MAX_CONTEXT



state = {} # dynamic global dict. (persist on verious HTTPS req.)

def load_ai_model():
    
    if not MODEL_DIR or not os.path.exists(MODEL_DIR):
        print(f"model directory invalid or missing : {MODEL_DIR}")
        return False
    #loading model onto ExLlamaV2
    print(f"Loading model from : {MODEL_DIR}...")
    
    try:
        
        config = ExLlamaV2Config(MODEL_DIR) #loads model's config
        config.prepare()
        config.max_seq_len = MAX_CONTEXT #sets max context
        
        #malloc for model
        model = ExLlamaV2(config)
        
        #lazy cache (sequential, to avoid fragmentaion errors) --> move to 8bit quantized cache 
        cache = ExLlamaV2Cache_Q8(model, max_seq_len=MAX_CONTEXT, lazy=True)
        
        #loads weights on VRAM
        def progress_callback(step, tot):
            print(f"\rLoading Weights... (Layer {step} of {tot})", end="", flush=True)
        model.load_autosplit(cache, callback=progress_callback) 
        print("\n")
        
        tokenizer = ExLlamaV2Tokenizer(config)
        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
        
        #makes dict.
        state["generator"] = generator
        state["tokenizer"] = tokenizer
        print("Model loaded on VRAM succesfully.")
        return True
    
    except torch.cuda.OutOfMemoryError:
        print(f"VRAM OoM during loading. Try lowering {MAX_CONTEXT}")
        #free corrupted malloc(s)
        torch.cuda.empty_cache() 
        return True
    
    except Exception as e:
        print(f"Failed to load model. -> {str(e)}")
        return True
    
    
def clear_ai_model():  
    #free() state (in wich the model was loaded in) when apllication shuts down, also free empty cache
    state.clear()
    torch.cuda.empty_cache()



