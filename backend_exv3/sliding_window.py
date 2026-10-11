import copy
from prompt_formatter import build_chatML_prompt
from config import sliding_window_trocate_buffer

def prune_messages_to_fit(messages: list, tokenizer, max_allowed_tokens: int) -> list:
    system_msgs = [copy.deepcopy(m) for m in messages if m.role == "system"]
    conversation_msgs = [copy.deepcopy(m) for m in messages if m.role != "system"]

    latest_msg = None
    latest_msg_len = 0
    sys_budget = max_allowed_tokens

    # 1. Reserve budget for the user's latest query
    if conversation_msgs:
        latest_msg = copy.deepcopy(conversation_msgs[-1])
        latest_msg_len = tokenizer.encode(latest_msg.content, encode_special_tokens=False).shape[-1]
        
        # Calculate budget for system messages (reserve space for the user query + formatting)
        sys_budget = max_allowed_tokens - latest_msg_len - 50 
        
        # Guarantee system messages (like tools/web search) get at least 1/3 of the context window
        if sys_budget < (max_allowed_tokens // 3):
            sys_budget = max_allowed_tokens // 3

    # 2. Forward iterate and truncate specific system messages that burst the budget
    if system_msgs:
        sys_prompt = build_chatML_prompt(system_msgs)
        sys_ids = tokenizer.encode(sys_prompt, encode_special_tokens=True)
        
        if sys_ids.shape[-1] > sys_budget:
            print(f"\n[Warning] System messages exceed budget ({sys_ids.shape[-1]} > {sys_budget}). Truncating system block.")
            
            retained_sys = []
            for i, sys_msg in enumerate(system_msgs):
                candidate_prompt = build_chatML_prompt(retained_sys + [sys_msg])
                candidate_len = tokenizer.encode(candidate_prompt, encode_special_tokens=True).shape[-1]
                
                if candidate_len > sys_budget:
                    # This specific system message blows the budget. Calculate exactly how much space is left for it.
                    base_prompt = build_chatML_prompt(retained_sys)
                    base_len = tokenizer.encode(base_prompt, encode_special_tokens=True).shape[-1]
                    
                    content_ids = tokenizer.encode(sys_msg.content, encode_special_tokens=False)
                    content_len = content_ids.shape[-1]
                    
                    formatting_overhead = candidate_len - base_len - content_len
                    allowed_sys_tokens = sys_budget - base_len - formatting_overhead - sliding_window_trocate_buffer
                    
                    if allowed_sys_tokens > 0:
                        truncated_ids = content_ids[:, :allowed_sys_tokens]
                        sys_msg.content = tokenizer.decode(truncated_ids)[0] + "\n\n...[SYSTEM: CONTENT TRUNCATED TO FIT WINDOW]..."
                        retained_sys.append(sys_msg)
                    else:
                        sys_msg.content = "...[SYSTEM: OMITTED TO FIT CONTEXT]..."
                        retained_sys.append(sys_msg)
                    
                    # Any system messages after this one are entirely omitted
                    for remaining_msg in system_msgs[i+1:]:
                        remaining_msg.content = "...[SYSTEM: OMITTED TO FIT CONTEXT]..."
                        retained_sys.append(remaining_msg)
                        
                    system_msgs = retained_sys
                    break
                else:
                    retained_sys.append(sys_msg)

    if not conversation_msgs:
        return system_msgs

    # 3. Re-evaluate the latest message with the properly pruned system block
    retained = [latest_msg]
    base_prompt = build_chatML_prompt(system_msgs + retained)
    base_len = tokenizer.encode(base_prompt, encode_special_tokens=True).shape[-1]
    
    if base_len > max_allowed_tokens:
        print(f"\n[Warning] Single user message exceeds {max_allowed_tokens} tokens. Truncating.")
        
        content_ids = tokenizer.encode(latest_msg.content, encode_special_tokens=False)
        content_len = content_ids.shape[-1]
        
        overhead = base_len - content_len
        allowed_content_tokens = max_allowed_tokens - overhead - sliding_window_trocate_buffer
        
        if allowed_content_tokens > 0:
            truncated_ids = content_ids[:, :allowed_content_tokens]
            latest_msg.content = tokenizer.decode(truncated_ids)[0] + "\n\n...[CONTENT TRUNCATED TO FIT WINDOW]..."
        else:
            latest_msg.content = "...[MESSAGE TOO LARGE, COMPLETELY OMITTED]..."
            
        return system_msgs + [latest_msg]

    # 4. Loop backward to append older conversation history until full
    for msg in reversed(conversation_msgs[:-1]):
        candidate = system_msgs + [msg] + retained
        test_prompt = build_chatML_prompt(candidate)
        test_len = tokenizer.encode(test_prompt, encode_special_tokens=True).shape[-1]
        
        if test_len > max_allowed_tokens:
            break
        retained.insert(0, msg)

    return system_msgs + retained