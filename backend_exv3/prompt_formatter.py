from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel
import datetime
import json


class Message(BaseModel):  
    role: Literal["system", "user", "assistant", "tool"]
    #the actual message
    content: str
    #claude code tools
    tool_calls: Optional[List[Dict[Any, Any]]] = None
    tool_call_id: Optional[str] = None

#fromatting JSON input into allowed format for the model (chatML)-->(https://web.archive.org/web/20230303120844/https://github.com/openai/openai-python/blob/main/chatml.md) (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)
def build_chatML_prompt(messages: List[Message]) -> str:
    now = datetime.datetime.now().strftime("%A, %B %d, %Y, %H:%M")

    system_parts = [
        f"",
        "You are a helpful AI assistant. Reply in the same language as the user's question.",
    ]

    conversation_parts = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "tool":
           
            conversation_parts.append(f"<|im_start|>tool\n{msg.content}<|im_end|>\n")
        else:
            
            extra = ""
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    
                    extra += f"\n<tool_call>\n{json.dumps({'name': func.get('name'), 'arguments': args}, ensure_ascii=False)}\n</tool_call>"
            
            conversation_parts.append(
                f"<|im_start|>{msg.role}\n{msg.content}{extra}<|im_end|>\n"
            )

    prompt = "<|im_start|>system\n" + "\n\n".join(system_parts) + "<|im_end|>\n"
    prompt += "".join(conversation_parts)
    prompt += "<|im_start|>assistant\n"

    return prompt