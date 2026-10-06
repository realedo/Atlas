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

    system_parts = [
        f"Current date and time: {now}",
        "You are a helpful AI assistant.",
    ]

    conversation_parts = []

    for msg in messages:
        if msg.role == "system":
            # Preserve injected instructions and web-search context
            system_parts.append(msg.content)
        else:
            conversation_parts.append(
                f"<|im_start|>{msg.role}\n"
                f"{msg.content}<|im_end|>\n"
            )

    prompt = (
        "<|im_start|>system\n"
        + "\n\n".join(system_parts)
        + "<|im_end|>\n"
    )

    prompt += "".join(conversation_parts)
    prompt += "<|im_start|>assistant\n"

    return prompt