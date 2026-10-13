import torch
import json
from prompt_formatter import Message
WARN = "\033[93m[Warning]\033[0m"

def inference_chunk(request_id: str, created_time: int, content=None, tool_call=None, finish_reason=None, role=None):
        delta = {}
        if role is not None:
            delta["role"] = role
            delta["content"] = ""
        if content is not None:
            delta["content"] = content
        if tool_call is not None:
            delta["tool_calls"] = [tool_call]
            
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": "exllamav3-local",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ]
        }
        
        
def declare_tools(data, safe_messages):
    tools = data.get("tools", [])
    if tools:
        tool_prompt = (
                "You have access to the following tools:\n\n"
                f"<tools>\n{json.dumps(tools, ensure_ascii=False, indent=2)}\n</tools>\n\n"
                "To use a tool, you MUST respond using the following XML format:\n"
                "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"arg_name\": \"value\"}}\n</tool_call>\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. You may ONLY use the tools explicitly listed above. Do not hallucinate or invent unlisted tools.\n"
                "2. Do not output any conversational text before or after the tool call XML block.\n"
                "3. Ensure your JSON tool arguments are valid. For multiline commands or scripts, carefully encode newlines as \\n."
            )
        safe_messages.insert(0, Message(role="system", content=tool_prompt))