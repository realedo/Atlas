from prompt_formatter import build_chatML_prompt


def prune_messages_to_fit(messages: list, tokenizer, max_allowed_tokens: int) -> list:

    system_msgs = [m for m in messages if m.role == "system"]
    conversation_msgs = [m for m in messages if m.role != "system"]

    # Always keep the latest turn
    if not conversation_msgs:
        return system_msgs

    # Start with system prompt + latest message
    retained = [conversation_msgs[-1]]

    # Work backwards adding messages until we run out of token budget
    for msg in reversed(conversation_msgs[:-1]):
        candidate = system_msgs + [msg] + retained
        test_prompt = build_chatML_prompt(candidate)
        test_len = tokenizer.encode(test_prompt, encode_special_tokens=True).shape[-1]
        
        if test_len > max_allowed_tokens:
            break
        retained.insert(0, msg)

    return system_msgs + retained