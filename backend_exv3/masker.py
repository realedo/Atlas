import json
import time


class ToolFilter:
    def __init__(self):
        self.in_tool = False
        self.buffer = ""
        self.tool_json_buffer = ""
        self.call_id = f"call_{int(time.time()*1000)}"

    def process_chunk(self, chunk: str) -> tuple[str, dict]:
        self.buffer += chunk
        text_out = ""
        tool_out = None
        
        while self.buffer:
            if self.in_tool:
                if "</tool_call>" in self.buffer:
                    tool_data, self.buffer = self.buffer.split("</tool_call>", 1)
                    self.tool_json_buffer += tool_data
                    
                    try:
                        
                        t_json = json.loads(self.tool_json_buffer.strip())
                        tool_out = {
                            "index": 0,
                            "id": self.call_id,
                            "type": "function",
                            "function": {
                                "name": t_json.get("name", ""),
                                "arguments": json.dumps(t_json.get("arguments", {}))
                            }
                        }
                    except Exception:
                        pass
                        
                    self.in_tool = False
                    self.tool_json_buffer = ""
                else:
                    self.tool_json_buffer += self.buffer
                    self.buffer = ""
                    break
            else:
                if "<tool_call>" in self.buffer:
                    before, self.buffer = self.buffer.split("<tool_call>", 1)
                    text_out += before
                    self.in_tool = True
                else:
                    safe_to_yield = self.buffer
                    # Blocca la scrittura se la stringa in arrivo sembra iniziare un tag <tool_call>
                    for i in range(1, 12): 
                        if self.buffer.endswith("<tool_call>"[:i]):
                            safe_to_yield = self.buffer[:-i]
                            self.buffer = self.buffer[-i:]
                            break
                    else:
                        self.buffer = ""
                        
                    if safe_to_yield:
                        text_out += safe_to_yield
                    break
                    
        return text_out, tool_out

    def flush(self) -> str:
        out = self.buffer if not self.in_tool else ""
        self.buffer = ""
        return out
    
    
    
    
    
class CoTFilter:
    def __init__(self):
        self.in_think = False
        self.thought_finished = False
        self.has_real_text = False
        self.buffer = ""

    
    def process_chunk(self, chunk: str) -> str:
       
        self.buffer += chunk
        output = ""

        while self.buffer:
            if self.in_think:
                if "</think>" in self.buffer:
                    _, self.buffer = self.buffer.split("</think>", 1)
                    self.in_think = False
                    self.thought_finished = True
                else:
                    for i in range(1, 8):
                        if self.buffer.endswith("</think>"[:i]):
                            self.buffer = self.buffer[-i:]
                            break
                    else:
                        self.buffer = ""
                    break
            else:
                if self.thought_finished or self.has_real_text:
                    output += self.buffer
                    self.buffer = ""
                    break
                    
                if "<think>" in self.buffer:
                    before, self.buffer = self.buffer.split("<think>", 1)
                    if before:
                        output += before
                        if before.strip():
                            self.has_real_text = True 
                            
                    if not self.has_real_text:
                        self.in_think = True
                else:
                    safe_to_yield = self.buffer
                    for i in range(1, 7):
                        if self.buffer.endswith("<think>"[:i]):
                            safe_to_yield = self.buffer[:-i]
                            self.buffer = self.buffer[-i:]
                            break
                    else:
                        self.buffer = ""
                        
                    if safe_to_yield:
                        output += safe_to_yield
                        if safe_to_yield.strip():
                            self.has_real_text = True
                    break
                    
        return output

    def flush(self) -> str:
        
        if self.buffer and not self.in_think:
            output = self.buffer
            self.buffer = ""
            return output
        return ""