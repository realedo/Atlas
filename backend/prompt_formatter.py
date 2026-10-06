from typing import List, Literal
import datetime
from pydantic import BaseModel

class Message(BaseModel):  #role + message
    role: Literal["system", "user", "assistant"]
    #the actual message
    content: str

#fromatting JSON input into allowed format for the model (chatML)-->(https://web.archive.org/web/20230303120844/https://github.com/openai/openai-python/blob/main/chatml.md) (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)
def build_chatML_prompt(messages: List[Message]) -> str:
   
    now = datetime.datetime.now().strftime("%A, %B %d, %Y, %H:%M")
       
    prompt = f"<|im_start|>system\nCurrent date and time: {now}\nYou are a helpful AI assistant.<|im_end|>\n"
    for msg in messages:
        #dont touch system fields
        if msg.role == "system":
            continue
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        
    prompt += "<|im_start|>assistant\n"
    return prompt