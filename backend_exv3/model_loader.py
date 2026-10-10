import torch
import os
import gc
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant
from config import MODEL_DIR, MAX_CONTEXT



state = {} # dynamic global dict. (persist on verious HTTPS req.)

def load_ai_model():
    
    if not MODEL_DIR or not os.path.exists(MODEL_DIR):
        print(f"model directory invalid or missing : {MODEL_DIR}")
        return False
    #loading model onto ExLlamaV3
    print(f"Loading model from : {MODEL_DIR}...")
    model = None
    cache = None
    tokenizer = None
    generator = None
    
    
    try:
        
        
        config = Config.from_directory(MODEL_DIR)
        model = Model.from_config(config)
        
        cache = Cache(
            model,
            max_num_tokens=MAX_CONTEXT,
            max_batch_size=1,
            layer_type=CacheLayer_quant,
            k_bits=8,
            v_bits=8,
        )
        
        model.load(progressbar=True)
        tokenizer = Tokenizer.from_config(config)
        
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            max_batch_size=1,
        )

        state.update({
            "config": config,
            "model": model,
            "cache": cache,
            "tokenizer": tokenizer,
            "generator": generator,
        })

        print("Model loaded successfully.")
        return True

        
    except torch.cuda.OutOfMemoryError:
        print(f"[Error] VRAM OoM during loading. Try lowering {MAX_CONTEXT}")
        #free corrupted malloc(s)
        torch.cuda.empty_cache() 
        return False
    
    except Exception as e:
        print(f"[Error] Failed to load model. -> {str(e)}")
        return False
    




def clear_ai_model():
    
    model = state.get("model")

    state.clear()

    if model is not None:
        try:
            model.unload()
        except Exception as exc:
            print(f"[WARNING] Model unload reported: {exc}")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Model resources released.")
