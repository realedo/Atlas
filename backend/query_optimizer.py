from exllamav2.generator import ExLlamaV2Sampler
from config import query_transform_temperature, query_transform_max_tokens
import torch
import json
import datetime

now = datetime.datetime.now().strftime("%A, %B %d, %Y")

def gen_optimized_query_to_search(generator, tokenizer, user_msg: str) -> str:
    
    #will set role as system and this as its scope
    sys_section=(
        "You are a model whose whole use case is taking the user prompt and generate a web search query\n"
        "Exctract 2/4 keywords from the user prompt and output a small quaery containing that will be used in a search engine\n"
        f"If you think its time sensitive, use datetime provided here: {now}"
        "Avoid any useless punctuation, special charachters if no needed. focus heavily on the keyword to compose the query.\n"
        
    )
    
    prompt = f"<|im_start|>system\n{sys_section}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    
    #tokenise the prompt
    input_ids = tokenizer.encode(prompt)
    
    
    try:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = query_transform_temperature
        
        generator.begin_stream(input_ids, settings)
        
        opt_query = ""
        for _ in range(query_transform_max_tokens):
            chunk, eos, _ = generator.stream()
            opt_query += chunk
            
            if eos:
                break
        return opt_query.strip().replace('"', '')
    
    except torch.cuda.OutOfMemoryError:
        #free memory
        torch.cuda.empty_cache()
        error_payload = {
                    "error": "VRAM OoM. Conversation context exeeded avalable memory."
                }
        return f"OoM while optimizing prompt for web search:\n{json.dumps(error_payload)}"
    
            #general error  
    except Exception as e:
        error_payload = {
                    "error": f"Internal inference error -> {str(e)}"
                }
        return f"Inference error while optimising prompt for web search:\n{json.dumps(error_payload)}"
        
                
            
    
    
    