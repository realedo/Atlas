import time
import re
from config import WEB_TOGGLE, max_url_scrape_len
from prompt_formatter import Message
from web_search import search_url, web_search
from query_optimizer import gen_optimized_query_to_search

#print() colors
INFO = "\033[92m[Info]\033[0m"
WARN = "\033[93m[Warning]\033[0m"
ERR = "\033[91m[Error]\033[0m"

url_pattern = re.compile(r'(https?://[^\s]+)')

def parse_search_result(result, max_scrape_len=max_url_scrape_len):
    
    try:
        # Validate expected fields exist
        if not isinstance(result, dict):
            return None
            
        required_fields = ['title', 'url', 'content']
        if not all(field in result for field in required_fields):
            return None
            
        # Limit content length defensively
        content = str(result.get('content', ''))[:max_scrape_len]
        return {
            'title': str(result.get('title', '')), 
            'url': str(result.get('url', '')), 
            'content': content
        }
    except Exception:
        return None

def add_web_context(last_user_message, inference_lock, generator, tokenizer, safe_messages):
    web_context = ""
    if WEB_TOGGLE and last_user_message.strip():
        print("Starting web search...\n")
        web_start_time = time.time()

        #checks if theres URLs in the prompt:
        try:
            urls_present = url_pattern.findall(last_user_message)
            if urls_present:
                url_to_search = urls_present[0]
                print(f"{INFO} Url ({url_to_search}) found in prompt, proceding to scrape from said url...\n")
                web_context = search_url(url_to_search)

            else:
                print(f"{WARN} No direct URLs found in prompt. generating a optimized search query...\n")

                with inference_lock: #cant inference both gen_optimized_query_to_search() and event_stream()
                    optimized_query = gen_optimized_query_to_search(
                        generator,
                        tokenizer,
                        last_user_message,
                    )

                print(f"{INFO}  Searching for: {optimized_query}\n")
                web_context = web_search(optimized_query)

        except Exception as e:
            print(f"{WARN} Failed web search: {str(e)}")
            web_context = ""

        print(f"{INFO}  Searched for {time.time()- web_start_time:.2f} seconds...\n")

    if web_context and not web_context.startswith(("Web search initialization failed:", "Web search is disabled")):
        #add web search
        safe_messages.insert(0,
             Message(
                 role="system",
                 content=("Your harness gives access to the following web search results. Use them when they are relevant to the user's question."
                          "Prioritize the supplied sources for current or time-sensitive facts, but do not assume every result is accurate."
                          "Do not invent facts or claim that a source supports something it does not. If the results are insufficient, say so."
                          "Always provide extensive, highly detailed, and comprehensive answers. Never give brief or single-sentence responses."
                          "Cite relevant sources if you absolutly deem necessary to, using their supplied URLs.\n\n" f"{web_context}"
                          "You are also using ExLlamaV3 inference engine on a tailored backend. the weights are from Qwen3.5(think) 10B 4bpw\n"
                          "Make sure to reason and use an appropiate amount of CoT tokens if the task implies reasoning")
             )
        )