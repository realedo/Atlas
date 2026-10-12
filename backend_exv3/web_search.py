import time
from ddgs import DDGS
import trafilatura
from config import (
    WEB_TOGGLE, 
    MX_RES, 
    max_url_scrape_len, 
    mx_trafilatura_per_prompt, 
    append_links, 
    append_images
)


INFO = "\033[92m[Info]\033[0m"
WARN = "\033[93m[Warning]\033[0m"
ERR = "\033[91m[Error]\033[0m"

# Cache the DDGS instance to prevent expensive session recreation
_ddgs_client = None

def get_ddgs():
    global _ddgs_client
    if _ddgs_client is None:
        _ddgs_client = DDGS()
    return _ddgs_client

def search_url(url: str, max_len: int = max_url_scrape_len) -> str:
    
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            downloaded_data = trafilatura.fetch_url(url)
            if downloaded_data:
                text = trafilatura.extract(
                    downloaded_data, 
                    include_links=append_links, 
                    include_images=append_images
                )
                if text:
                    print(f"{INFO} Extracted context from given URL ({url})\n")
                    if len(text) > max_len:
                        return text[:max_len] + f"\n\n...[Article has been cut due to being longer than max len ({max_len})]"
                    return text
            
            print(f"{WARN} No content extracted from {url} on attempt {attempt + 1}")
            
        except Exception as e:
            print(f"{WARN} Error fetching data from {url} (Attempt {attempt + 1}/{max_retries}): {str(e)}")
        
        # Exponential backoff before retrying
        if attempt < max_retries - 1:
            time.sleep(base_delay * (2 ** attempt))

    return "Could not fetch data from provided url."

def web_search(query: str, max_res: int = MX_RES) -> str:
    
    if not WEB_TOGGLE:
        return "Web search is disabled."
    
    query = query.strip()
    if not query:
        return "Web search requires a non-empty query."
        
    sections = []
    seen_urls = set()
    
    try:
        ddgs = get_ddgs()
        
        # --- Text Search ---
        try:
            results = list(ddgs.text(query, max_results=max_res) or [])
            if results:
                lines = ["Web search results:"]
                for i, result in enumerate(results, start=1):
                    url = result.get('href', '')
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    lines.append(
                        f"{i}. Title: {result.get('title', 'Untitled')}\n"
                        f"   Snippet: {result.get('body', 'No snippet available')}\n"
                        f"   URL: {url}"
                    )
                
                # Extract content from the top MX results
                extract_count = min(mx_trafilatura_per_prompt, len(results))
                extracted = 0
                idx = 0
                
                while extracted < extract_count and idx < len(results):
                    url_to_scrape = results[idx].get('href')
                    if url_to_scrape:
                        print(f"{INFO} Attempting to extract data from result {idx+1}\n")
                        lines.append("\nData extracted from the page:\n")
                        lines.append(search_url(url_to_scrape))
                        extracted += 1
                    idx += 1
                        
                sections.append("\n".join(lines))
                
        except Exception as e:
            print(f"{WARN} Text search failed: {str(e)}")
            sections.append(f"Text search failed: {str(e)}")

        # --- News Search ---
        try:
            news = list(ddgs.news(query, max_results=max_res) or [])
            if news:
                lines = ["\nNews search results:"]
                for i, result in enumerate(news, start=1):
                    url = result.get('url', result.get('href', ''))
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    lines.append(
                        f"{i}. Title: {result.get('title', 'Untitled')}\n"
                        f"   Snippet: {result.get('body', 'No snippet available')}\n"
                        f"   URL: {url}\n"
                        f"   Date: {result.get('date', 'Unknown')}"
                    )
                sections.append("\n".join(lines))
                
        except Exception as e:
            print(f"{WARN} News search failed: {str(e)}")
            sections.append(f"News search failed: {str(e)}")

    except Exception as e:
        return f"Web search initialization failed: {e}"

    if not sections:
        return "No web or news results found for this query."

    return f"Search query: {query}\n\n" + "\n\n".join(sections)