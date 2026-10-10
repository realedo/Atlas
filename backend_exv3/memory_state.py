from typing import List
from prompt_formatter import Message
from config import max_memory_turns

class ConversationMemory:
    def __init__(self, max_turns=max_memory_turns):
        
        self.history: List[Message] = []
        self.max_turns = max_turns

    def add_interaction(self, user_text: str, assistant_text: str):
        #if not None, save last convo (user and assistant no system!)
        if user_text.strip() and assistant_text.strip():
            self.history.append(Message(role="user", content=user_text))
            self.history.append(Message(role="assistant", content=assistant_text))
            
            #if convo longer than max_turns, delete oldest append new one
            if len(self.history) > self.max_turns * 2:
                self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[Message]:
        return self.history

    def clear(self):
        """Svuota la memoria della chat."""
        self.history = []

#memory importable on main code.
chat_state = ConversationMemory(max_turns=4)