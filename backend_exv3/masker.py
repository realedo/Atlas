import json
import time
import re

class ToolFilter:
    def __init__(self):
        self.in_tool = False
        self.buffer = ""
        self.tool_json_buffer = ""
        self.call_id = f"call_{int(time.time()*1000)}"
        
    def _generate_id(self):
        
        return f"call_{int(time.time()*1000000)}"

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
                        raw_json_str = self.tool_json_buffer.strip()
                        
                        try:
                            # 1. Try parsing exactly as provided. strict=False allows 
                            # literal unescaped newlines/tabs inside code blocks.
                            t_json = json.loads(raw_json_str, strict=False)
                        except json.JSONDecodeError:
                            # 2. Fallback: Safely escape invalid backslashes (like \U in Windows paths) 
                            # without corrupting already valid escapes or double backslashes.
                            def fix_escapes(m):
                                text = m.group(0)
                                # If it's a valid JSON escape sequence, leave it alone
                                if text in ['\\"', '\\\\', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t']:
                                    return text
                                if text.startswith('\\u'):
                                    return text
                                # If it's invalid (like \U or \D), add a second backslash
                                return '\\\\' + text[1:]
                                
                            sanitized = re.sub(r'\\.', fix_escapes, raw_json_str)
                            t_json = json.loads(sanitized, strict=False)

                        tool_out = {
                            "index": 0,
                            "id": self.call_id,
                            "type": "function",
                            "function": {
                                "name": t_json.get("name", ""),
                                "arguments": json.dumps(t_json.get("arguments", {}))
                            }
                        }
                        self.call_id = self._generate_id()
                        
                        
                    except Exception as e:
                        print(f"\n\033[91m[Error]\033[0m Failed to parse tool JSON: {str(e)}")
                        print(f"Raw buffer: {self.tool_json_buffer.strip()}")
                        
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
                        # Clean out rogue XML tags while preserving tool_call tags
                        cleaned_before = re.sub(r'<(?!/?tool_call>)[^>]+>', '', before)
                        if cleaned_before.strip():
                            output += cleaned_before
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
                        # Filter out stray unhandled tags while preserving tool_call tags
                        cleaned_yield = re.sub(r'<(?!/?tool_call>)[^>]+>', '', safe_to_yield)
                        if cleaned_yield.strip():
                            output += cleaned_yield
                            self.has_real_text = True
                    break
                
        return output

    def flush(self, force=False) -> str:
        if self.buffer and (not self.in_think or force):
            output = self.buffer
            self.buffer = ""
            return output
        return ""