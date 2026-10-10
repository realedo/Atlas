from exllamav3.generator.sampler import ComboSampler
from exllamav3.generator import Job
from config import query_transform_temperature, query_transform_max_tokens
import torch
import json
import datetime


def gen_optimized_query_to_search(generator, tokenizer, user_msg: str) -> str:
    
    now = datetime.datetime.now().strftime("%A, %B %d, %Y")
    
    #will set role as system and this as its scope
    sys_section=(
        "You are a model whose whole use case is taking the user prompt and generate a web search query\n"
        "Exctract 2/4 keywords from the user prompt and output a small quaery containing that will be used in a search engine\n"
        f"If you think its time sensitive, use datetime provided here: {now}"
        "Avoid any useless punctuation, special charachters if no needed. focus heavily on the keyword to compose the query.\n"
        
    )
    
    #prompt with CoT alredy done (i dont want it to make more of them)
    prompt = f"<|im_start|>system\n{sys_section}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\nSkipping thought process.\n</think>\n"
    
    #tokenise the prompt
    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
    
    try:
       
        sampler = ComboSampler(
            temperature=float(query_transform_temperature)
        )
        
        #end conditions
        stop_conditions = ["<|im_end|>"]
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_conditions.append(int(eos_token_id))

        
        job = Job(
            input_ids=input_ids,
            max_new_tokens=query_transform_max_tokens,
            sampler=sampler,
            stop_conditions=stop_conditions,
            decode_special_tokens=False,
        )

        #inference
        generator.enqueue(job)
        
        opt_query = ""
        while generator.num_remaining_jobs():
            results = generator.iterate()
            
            for result in results:
                if result.get("stage") == "error":
                    raise result["error"]
                    
                if result.get("stage") != "streaming":
                    continue
                    
                chunk = result.get("text", "")
                if chunk:
                    opt_query += chunk
                    
                if result.get("eos"):
                    break 

            
            if not generator.num_remaining_jobs():
                break

        return opt_query.strip().replace('"', '')
    
    except torch.cuda.OutOfMemoryError:
        #free memory
        if torch.cuda.is_available():
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