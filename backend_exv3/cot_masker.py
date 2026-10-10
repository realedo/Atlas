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